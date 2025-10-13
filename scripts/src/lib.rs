

use serde_json::{json, Value};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

#[derive(Debug, Clone)]
pub struct OHLCV {
    pub ts: Vec<i64>,        
    pub close: Vec<f64>,
    pub high: Option<Vec<f64>>,
    pub low: Option<Vec<f64>>,
    pub signal: Vec<i8>,     
}

#[derive(Debug, Clone)]
struct Position {
    symbol: String,
    qty: f64,
    buy_price: f64,
    buy_time: i64,           
    sell_time: i64,          
    sell_time_int: i64,      
    commission_buy: f64,
    exec_price: Option<f64>,
    exec_time: Option<i64>,
    exec_time_int: Option<i64>,
    exit_reason: Option<String>,
    closed: bool,
}

fn upper_bound_int(slice: &Vec<i64>, target: i64) -> usize {
    
    let mut lo = 0usize;
    let mut hi = slice.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        if slice[mid] <= target {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

fn get_price_at_int(
    sym: &str,
    t: i64,
    sym_data: &HashMap<String, OHLCV>,
    ts_index_map_by_sym_int: &HashMap<String, HashMap<i64, usize>>,
) -> Option<f64> {
    if let Some(d) = sym_data.get(sym) {
        if let Some(idx_map) = ts_index_map_by_sym_int.get(sym) {
            if let Some(&idx) = idx_map.get(&t) {
                return Some(d.close[idx]);
            }
        }
        
        let idx = if d.ts.is_empty() {
            None
        } else {
            let pos = upper_bound_int(&d.ts, t);
            if pos == 0 {
                None
            } else {
                Some(pos - 1)
            }
        };
        if let Some(i) = idx {
            return Some(d.close[i]);
        }
    }
    None
}

pub fn run_grid_backtest(
    ohlcv_arrays: HashMap<String, OHLCV>,
    sell_after: usize,
    initial_balance: f64,
    order_amount: f64,
    tp_pct: f64,
    sl_pct: f64,
    comi_pct: f64,
) -> Value {
    // config inicial
    let comi_factor = comi_pct / 100.0;
    let mut cash = initial_balance;
    let mut num_signals_executed = 0usize;

    // preparar sym_data (ya lo recibimos en ohlcv_arrays)
    let symbols: Vec<String> = ohlcv_arrays.keys().cloned().collect();
    let mut sym_data: HashMap<String, OHLCV> = HashMap::new();
    for (k, v) in ohlcv_arrays.into_iter() {
        sym_data.insert(k, v);
    }

    // signals_by_time: ts_int -> Vec<(sym, idx)>
    let mut signals_by_time: HashMap<i64, Vec<(String, usize)>> = HashMap::new();
    for sym in &symbols {
        let d = &sym_data[sym];
        for (idx, &s) in d.signal.iter().enumerate() {
            if s != 0 {
                let t_int = d.ts[idx];
                signals_by_time
                    .entry(t_int)
                    .or_default()
                    .push((sym.clone(), idx));
            }
        }
    }

    // all timestamps
    let mut all_ts_set: Vec<i64> = Vec::new();
    {
        use std::collections::BTreeSet;
        let mut sset = BTreeSet::new();
        for d in sym_data.values() {
            for &t in &d.ts {
                sset.insert(t);
            }
        }
        all_ts_set = sset.into_iter().collect();
    }
    let mut all_timestamps_int = all_ts_set.clone(); // sorted
    let all_timestamps_dt = all_timestamps_int.clone(); 

   
    let mut trades: HashMap<String, Vec<f64>> = HashMap::new();
    let mut trade_times: HashMap<String, Vec<i64>> = HashMap::new();
    for s in &symbols {
        trades.insert(s.clone(), Vec::new());
        trade_times.insert(s.clone(), Vec::new());
    }

    
    let mut trade_log_rows: Vec<Value> = Vec::new();

    // sim balance
    let mut sim_timestamps: Vec<i64> = Vec::new();
    let mut sim_balances: Vec<f64> = Vec::new();

    let mut open_heap: BinaryHeap<Reverse<(i64, usize, usize)>> = BinaryHeap::new();
    let mut positions_store: Vec<Position> = Vec::new();
    let mut counter: usize = 0usize;
    let symbol_order: HashMap<String, usize> = symbols
        .iter()
        .enumerate()
        .map(|(i, s)| (s.clone(), i))
        .collect();

    let ts_index_map_by_sym_int: HashMap<String, HashMap<i64, usize>> = {
        let mut m = HashMap::new();
        for (sym, d) in sym_data.iter() {
            let mut map = HashMap::new();
            for (idx, &t) in d.ts.iter().enumerate() {
                map.insert(t, idx);
            }
            m.insert(sym.clone(), map);
        }
        m
    };

   
    let mut close_position = |pos_idx: usize, exec_time: i64, exec_price: f64, exit_reason: &str| {
        let pos = &mut positions_store[pos_idx];
        if pos.closed {
            return;
        }
    
        let qty = pos.qty;
        let buy_price = pos.buy_price;
        let commission_buy = pos.commission_buy;
        let commission_sell = if comi_factor != 0.0 {
            (qty * exec_price) * comi_factor
        } else {
            0.0
        };
    
        cash += qty * exec_price - commission_sell;
        let profit = (exec_price - buy_price) * qty - commission_buy - commission_sell;
    
        if let Some(vec) = trades.get_mut(&pos.symbol) {
            vec.push(profit);
        }
        if let Some(vect) = trade_times.get_mut(&pos.symbol) {
            vect.push(exec_time);
        }
    
        trade_log_rows.push(json!({
            "symbol": pos.symbol,
            "buy_time": pos.buy_time,
            "buy_price": pos.buy_price,
            "sell_time": exec_time,
            "sell_price": exec_price,
            "qty": qty,
            "profit": profit,
            "exit_reason": exit_reason,
            "commission_buy": commission_buy,
            "commission_sell": commission_sell
        }));
    
        pos.closed = true; 
    };


    
    for &t_int in &all_timestamps_int {
        
        while let Some(Reverse((top_time, _top_counter, pos_idx))) = open_heap.peek().cloned() {
            if top_time > t_int {
                break;
            }
            
            let Reverse((_time, _ct, pos_idx)) = open_heap.pop().unwrap();
            
            if positions_store[pos_idx].closed {
                continue;
            }
            let pos = &positions_store[pos_idx];
            if let (Some(exec_price), Some(exec_time_int)) = (pos.exec_price, pos.exec_time_int) {
                if exec_time_int <= t_int {
                    
                    let exec_time = pos.exec_time.unwrap();
                    close_position(pos_idx, exec_time, exec_price, pos.exit_reason.as_deref().unwrap_or("EXEC"));
                    
                    continue;
                }
            }
            
            let sym = pos.symbol.clone();
            let sell_ts_int = pos.sell_time_int;
            let maybe_price = get_price_at_int(&sym, sell_ts_int, &sym_data, &ts_index_map_by_sym_int);
            if let Some(exec_price) = maybe_price {
                let exec_time_dt = sell_ts_int;
                close_position(pos_idx, exec_time_dt, exec_price, "SELL_AFTER");
            } else {
                
                let d = &sym_data[&sym];
                let exec_price = *d.close.last().unwrap_or(&0.0);
                let last_time_dt = *d.ts.last().unwrap_or(&sell_ts_int);
                close_position(pos_idx, last_time_dt, exec_price, "FORCED_LAST");
            }
            
        }

        
        let mut any_open = false;
        let mut positions_value = 0.0f64;
        for Reverse((_time, _ct, pos_idx)) in open_heap.clone().into_sorted_vec() {
            if !positions_store[pos_idx].closed {
                any_open = true;
                let pos = &positions_store[pos_idx];
                if let Some(price) =
                    get_price_at_int(&pos.symbol, t_int, &sym_data, &ts_index_map_by_sym_int)
                {
                    positions_value += pos.qty * price;
                }
            }
        }
        if any_open {
            sim_timestamps.push(t_int);
            sim_balances.push(cash + positions_value);
            continue;
        }

        
        if let Some(events) = signals_by_time.get(&t_int) {
            
            let mut events_sorted = events.clone();
            events_sorted.sort_by_key(|(s, _idx)| symbol_order.get(s).cloned().unwrap_or(usize::MAX));
            for (sym, buy_idx) in events_sorted {
                if cash < order_amount {
                    break;
                }
                let d = &sym_data[&sym];
                let price_t = d.close[buy_idx];
                let qty = order_amount / price_t;
                let commission_buy = if comi_factor != 0.0 { order_amount * comi_factor } else { 0.0 };
                cash -= order_amount + commission_buy;
                num_signals_executed += 1;

                let sell_idx = std::cmp::min(buy_idx + sell_after, d.ts.len().saturating_sub(1));
                let sell_time_dt = d.ts[sell_idx];
                let sell_time_int = d.ts[sell_idx];

                let tp_price = if tp_pct != 0.0 { price_t * (1.0 + tp_pct / 100.0) } else { f64::INFINITY };
                let sl_price = if sl_pct != 0.0 { price_t * (1.0 - sl_pct / 100.0) } else { f64::NEG_INFINITY };

                let mut position = Position {
                    symbol: sym.clone(),
                    qty,
                    buy_price: price_t,
                    buy_time: t_int,
                    sell_time: sell_time_dt,
                    sell_time_int,
                    commission_buy,
                    exec_price: None,
                    exec_time: None,
                    exec_time_int: None,
                    exit_reason: None,
                    closed: false,
                };

                
                let mut intravela_detected = false;
                if d.high.is_some() && d.low.is_some() && (tp_pct != 0.0 || sl_pct != 0.0) {
                    let high = d.high.as_ref().unwrap();
                    let low = d.low.as_ref().unwrap();
                    let start = buy_idx + 1;
                    let end = sell_idx;
                    if end >= start && end < high.len() {
                        // slices
                        let high_slice = &high[start..=end];
                        let low_slice = &low[start..=end];

                        
                        let mut tp_first: Option<usize> = None;
                        if tp_price.is_finite() {
                            for (i, &v) in high_slice.iter().enumerate() {
                                if v >= tp_price {
                                    tp_first = Some(start + i);
                                    break;
                                }
                            }
                        }

                        let mut sl_first: Option<usize> = None;
                        if sl_price.is_finite() {
                            for (i, &v) in low_slice.iter().enumerate() {
                                if v <= sl_price {
                                    sl_first = Some(start + i);
                                    break;
                                }
                            }
                        }

                        if let (Some(tp_i), Some(sl_i)) = (tp_first, sl_first) {
                            if sl_i <= tp_i {
                                let chosen_idx = sl_i;
                                position.exec_price = Some(sl_price);
                                position.exec_time = Some(d.ts[chosen_idx]);
                                position.exec_time_int = Some(d.ts[chosen_idx]);
                                position.exit_reason = Some("SL".to_string());
                                intravela_detected = true;
                            } else {
                                let chosen_idx = tp_i;
                                position.exec_price = Some(tp_price);
                                position.exec_time = Some(d.ts[chosen_idx]);
                                position.exec_time_int = Some(d.ts[chosen_idx]);
                                position.exit_reason = Some("TP".to_string());
                                intravela_detected = true;
                            }
                        } else if let Some(sl_i) = sl_first {
                            let chosen_idx = sl_i;
                            position.exec_price = Some(sl_price);
                            position.exec_time = Some(d.ts[chosen_idx]);
                            position.exec_time_int = Some(d.ts[chosen_idx]);
                            position.exit_reason = Some("SL".to_string());
                            intravela_detected = true;
                        } else if let Some(tp_i) = tp_first {
                            let chosen_idx = tp_i;
                            position.exec_price = Some(tp_price);
                            position.exec_time = Some(d.ts[chosen_idx]);
                            position.exec_time_int = Some(d.ts[chosen_idx]);
                            position.exit_reason = Some("TP".to_string());
                            intravela_detected = true;
                        }
                    }
                }

                
                let pos_idx = positions_store.len();
                positions_store.push(position);
                if intravela_detected {
                    let p = &positions_store[pos_idx];
                    if let Some(exec_time_int) = p.exec_time_int {
                        open_heap.push(Reverse((exec_time_int, counter, pos_idx)));
                        counter += 1;
                    } else {
                        // fallback to sell_time
                        open_heap.push(Reverse((p.sell_time_int, counter, pos_idx)));
                        counter += 1;
                    }
                } else {
                    let p = &positions_store[pos_idx];
                    open_heap.push(Reverse((p.sell_time_int, counter, pos_idx)));
                    counter += 1;
                }
            }
        }

        
        let mut positions_value2 = 0.0f64;
        for Reverse((_time, _ct, pos_idx)) in open_heap.clone().into_sorted_vec() {
            if !positions_store[pos_idx].closed {
                let pos = &positions_store[pos_idx];
                if let Some(price) =
                    get_price_at_int(&pos.symbol, t_int, &sym_data, &ts_index_map_by_sym_int)
                {
                    positions_value2 += pos.qty * price;
                }
            }
        }
        sim_timestamps.push(t_int);
        sim_balances.push(cash + positions_value2);
    }

    
    while let Some(Reverse((_time, _ct, pos_idx))) = open_heap.pop() {
        if positions_store[pos_idx].closed {
            continue;
        }
        let sym = positions_store[pos_idx].symbol.clone();
        let d = &sym_data[&sym];
        if let Some(exec_price) = positions_store[pos_idx].exec_price {
            let exec_time = positions_store[pos_idx].exec_time.unwrap();
            close_position(pos_idx, exec_time, exec_price, positions_store[pos_idx].exit_reason.as_deref().unwrap_or("EXEC"));
        } else {
            let sell_ts_int = positions_store[pos_idx].sell_time_int;
            if let Some(exec_price) = get_price_at_int(&sym, sell_ts_int, &sym_data, &ts_index_map_by_sym_int) {
                let exec_time_dt = sell_ts_int;
                close_position(pos_idx, exec_time_dt, exec_price, "SELL_AFTER");
            } else {
                let exec_price = *d.close.last().unwrap_or(&0.0);
                let last_time_dt = *d.ts.last().unwrap_or(&sell_ts_int);
                close_position(pos_idx, last_time_dt, exec_price, "FORCED_LAST");
            }
        }
        
    }

    
    let final_balance = cash;
    let mut all_trades: Vec<f64> = Vec::new();
    for s in &symbols {
        if let Some(v) = trades.get(s) {
            for &p in v { all_trades.push(p); }
        }
    }
    let num_trades = all_trades.len();
    let proportion_winners = if num_trades > 0 {
        let winners = all_trades.iter().filter(|&&x| x > 0.0).count();
        (winners as f64) / (num_trades as f64)
    } else {
        f64::NAN
    };

    
    let mut ts_index_map: HashMap<i64, usize> = HashMap::new();
    for (i, &t) in all_timestamps_int.iter().enumerate() {
        ts_index_map.insert(t, i);
    }

    let mut max_dd_by_symbol: HashMap<String, f64> = HashMap::new();
    let mut final_balance_by_symbol: HashMap<String, f64> = HashMap::new();
    let mut equity_by_symbol: HashMap<String, Vec<f64>> = HashMap::new();

    for sym in &symbols {
        let mut profits_series = vec![0.0f64; all_timestamps_int.len()];
        if let Some(trs) = trades.get(sym) {
            if let Some(tts) = trade_times.get(sym) {
                for (&profit, &t_close) in trs.iter().zip(tts.iter()) {
                    if let Some(&idx) = ts_index_map.get(&t_close) {
                        profits_series[idx] += profit;
                    } else {
                        
                        let pos = upper_bound_int(&all_timestamps_int, t_close);
                        let idx = if pos == 0 { None } else { Some(pos - 1) };
                        if let Some(i) = idx {
                            profits_series[i] += profit;
                        }
                    }
                }
            }
        }
        
        let mut equity: Vec<f64> = Vec::with_capacity(profits_series.len());
        let mut acc = initial_balance;
        for p in profits_series.iter() {
            acc += *p;
            equity.push(acc);
        }
        equity_by_symbol.insert(sym.clone(), equity.clone());
        if equity.len() > 0 {
            
            let mut cummax = vec![0.0f64; equity.len()];
            let mut m = std::f64::NEG_INFINITY;
            for (i, &v) in equity.iter().enumerate() {
                if v > m { m = v; }
                cummax[i] = m;
            }
            
            let mut drawdowns = vec![0.0f64; equity.len()];
            for i in 0..equity.len() {
                let denom = if cummax[i] == 0.0 { 1.0 } else { cummax[i] };
                drawdowns[i] = (cummax[i] - equity[i]) / denom;
            }
            let max_dd = drawdowns.iter().cloned().fold(f64::NAN, |a, b| if a.is_nan() { b } else { a.max(b) });
            max_dd_by_symbol.insert(sym.clone(), max_dd);
            final_balance_by_symbol.insert(sym.clone(), *equity.last().unwrap());
        } else {
            max_dd_by_symbol.insert(sym.clone(), 0.0);
            final_balance_by_symbol.insert(sym.clone(), initial_balance);
        }
    }

    let sim_values: Vec<f64> = if sim_balances.is_empty() {
        vec![initial_balance]
    } else {
        sim_balances.clone()
    };

    
    let mut cummax = vec![0.0f64; sim_values.len()];
    let mut m = std::f64::NEG_INFINITY;
    for (i, &v) in sim_values.iter().enumerate() {
        if v > m { m = v; }
        cummax[i] = m;
    }
    let mut drawdowns = vec![0.0f64; sim_values.len()];
    for i in 0..sim_values.len() {
        let denom = if cummax[i] == 0.0 { 1.0 } else { cummax[i] };
        drawdowns[i] = (cummax[i] - sim_values[i]) / denom;
    }
    let max_dd_portfolio = drawdowns.iter().cloned().fold(f64::NAN, |a, b| if a.is_nan() { b } else { a.max(b) });

    
    let compute_annualized_sharpe_from_equity = |equity_arr: &Vec<f64>, time_index_dt: &Vec<i64>| -> f64 {
        if equity_arr.len() < 2 {
            return f64::NAN;
        }
        
        let mut returns: Vec<f64> = Vec::new();
        for i in 1..equity_arr.len() {
            if equity_arr[i-1] != 0.0 {
                returns.push((equity_arr[i] / equity_arr[i-1]) - 1.0);
            }
        }
        returns.retain(|x| x.is_finite());
        if returns.is_empty() {
            return f64::NAN;
        }
        
        let median_delta_s = if time_index_dt.len() >= 2 {
            let mut deltas_s: Vec<f64> = Vec::new();
            for i in 1..time_index_dt.len() {
                let d_ns = (time_index_dt[i] as f64) - (time_index_dt[i-1] as f64);
                let d_s = d_ns / 1e9;
                if d_s > 0.0 {
                    deltas_s.push(d_s);
                }
            }
            if deltas_s.is_empty() {
                24.0*3600.0
            } else {
                deltas_s.sort_by(|a,b| a.partial_cmp(b).unwrap());
                let mid = deltas_s.len()/2;
                if deltas_s.len() % 2 == 1 {
                    deltas_s[mid]
                } else {
                    (deltas_s[mid-1] + deltas_s[mid]) / 2.0
                }
            }
        } else {
            24.0*3600.0
        };
        let periods_per_year = (365.0 * 24.0 * 3600.0) / median_delta_s;
        let mean_periodic = returns.iter().sum::<f64>() / (returns.len() as f64);
        let var = returns.iter().map(|r| (r - mean_periodic)*(r - mean_periodic)).sum::<f64>() / (returns.len() as f64);
        let std_periodic = var.sqrt();
        if !std_periodic.is_finite() || std_periodic == 0.0 {
            return f64::NAN;
        }
        let annualized_mean = mean_periodic * periods_per_year;
        let annualized_std = std_periodic * periods_per_year.sqrt();
        if annualized_std == 0.0 {
            return f64::NAN;
        }
        annualized_mean / annualized_std
    };

    let mut sharpe_by_symbol: HashMap<String, f64> = HashMap::new();
    for sym in &symbols {
        let equity_opt = equity_by_symbol.get(sym);
        let equity: Vec<f64> = match equity_opt {
            Some(eq) => eq.clone(),
            None => Vec::new(),
        };
        let sharpe = compute_annualized_sharpe_from_equity(&equity, &all_timestamps_dt);
        let sharpe = compute_annualized_sharpe_from_equity(equity, &all_timestamps_dt);
        sharpe_by_symbol.insert(sym.clone(), sharpe);
    }

    
    let sim_ts_arr = if sim_timestamps.len() >= 1 { sim_timestamps.clone() } else { all_timestamps_dt.clone() };
    let sharpe_portfolio = compute_annualized_sharpe_from_equity(&sim_values, &sim_ts_arr);

    
    let mut results_map = serde_json::Map::new();

    for sym in &symbols {
        let tr = trades.get(sym).cloned().unwrap_or_default();
        let num_signals = tr.len();
        let proportion = if num_signals == 0 { Value::Null } else {
            let winners = tr.iter().filter(|&&x| x > 0.0).count();
            Value::from((winners as f64) / (num_signals as f64))
        };
        let obj = json!({
            "df": Value::Null,
            "trades": tr,
            "final_balance": final_balance_by_symbol.get(sym).cloned().unwrap_or(initial_balance),
            "num_signals": num_signals,
            "proportion_winners": proportion,
            "max_dd": max_dd_by_symbol.get(sym).cloned().unwrap_or(0.0),
            "sharpe": sharpe_by_symbol.get(sym).cloned().unwrap_or(f64::NAN)
        });
        results_map.insert(sym.clone(), obj);
    }

    
    let trade_log_value = Value::Array(trade_log_rows);
    let sim_balance_history = json!({
        "timestamp": sim_timestamps,
        "balance": sim_values
    });

    let portfolio_obj = json!({
        "df": Value::Null,
        "trades": all_trades,
        "final_balance": final_balance,
        "num_signals": num_signals_executed,
        "proportion_winners": if num_trades>0 { Value::from(proportion_winners) } else { Value::Null },
        "max_dd": max_dd_portfolio,
        "sim_balance_history": sim_balance_history,
        "trade_log": trade_log_value,
        "sharpe": sharpe_portfolio
    });

    results_map.insert("__PORTFOLIO__".to_string(), portfolio_obj);

    Value::Object(results_map)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn smoke_test() {
        
        let mut ohlcv: HashMap<String, OHLCV> = HashMap::new();
        let ts = vec![1i64, 2, 3, 4, 5].into_iter().map(|x| x * 1_000_000_000).collect::<Vec<_>>();
        let close = vec![10.0, 10.5, 11.0, 10.8, 11.2];
        let high = close.clone();
        let low = close.clone();
        let signal = vec![0, 1, 0, 0, 0];
        ohlcv.insert("SYM".to_string(), OHLCV {
            ts: ts.clone(),
            close,
            high: Some(high),
            low: Some(low),
            signal,
        });
        let res = run_grid_backtest(ohlcv, 2, 10000.0, 100.0, 0.0, 0.0, 0.05);
        assert!(res.get("SYM").is_some());
        assert!(res.get("__PORTFOLIO__").is_some());
    }
}

use pyo3::prelude::*;
use pyo3::types::PyDict;


#[pyfunction]
fn run_grid_backtest_py(
    ohlcv_dict: &PyDict,
    sell_after: usize,
    initial_balance: f64,
    order_amount: f64,
    tp_pct: f64,
    sl_pct: f64,
    comi_pct: f64
) -> PyResult<String> {
    
    let mut ohlcv: HashMap<String, OHLCV> = HashMap::new();
    for (key, value) in ohlcv_dict.iter() {
        let sym: String = key.extract()?;
        let v = value.downcast::<PyDict>()?;
        // extraer campos, con errores claros si falta alguno
        let ts_any = v.get_item("ts").ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("ts missing"))?;
        let ts: Vec<i64> = ts_any.extract()?;
        let close_any = v.get_item("close").ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("close missing"))?;
        let close: Vec<f64> = close_any.extract()?;
        let signal_any = v.get_item("signal")?;
        let signal: Vec<i8> = signal_any.extract()?;

        let high = match v.get_item("high") {
            Ok(x) => Some(x.extract::<Vec<f64>>()?),
            Err(_) => None,
        };

        let low = match v.get_item("low") {
            Ok(Some(x)) => Some(x.extract::<Vec<f64>>()?),
            Ok(None) => None,
            Err(_) => None,
        };
        ohlcv.insert(sym, OHLCV { ts, close, high, low, signal });
    }

    let result_value = run_grid_backtest(ohlcv, sell_after, initial_balance, order_amount, tp_pct, sl_pct, comi_pct);
    
    match serde_json::to_string(&result_value) {
        Ok(s) => Ok(s),
        Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("serde_json error: {}", e))),
    }
}

#[pymodule]
fn rust_backtest(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_grid_backtest_py, m)?)?;
    Ok(())
}
