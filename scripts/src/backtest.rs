use std::collections::{HashMap, BinaryHeap};
use std::cmp::Ordering;
use serde::{Serialize, Deserialize};

// ============================
// Estructuras de datos
// ============================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OhlcvData {
    pub ts: Vec<i64>,           
    pub close: Vec<f64>,
    pub high: Option<Vec<f64>>,
    pub low: Option<Vec<f64>>,
    pub signal: Vec<bool>,
}

#[derive(Debug, Clone)]
struct SymbolData {
    ts: Vec<i64>,
    close: Vec<f64>,
    high: Option<Vec<f64>>,
    low: Option<Vec<f64>>,
    signal: Vec<bool>,
    len: usize,
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

// Heap auxiliar
#[derive(Clone)]
struct HeapItem {
    time: i64,
    counter: usize,
    position: Position,
}

impl PartialEq for HeapItem {
    fn eq(&self, other: &Self) -> bool {
        self.time == other.time
    }
}

impl Eq for HeapItem {}

impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        other.time.cmp(&self.time).then_with(|| other.counter.cmp(&self.counter))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeLog {
    pub symbol: Vec<String>,
    pub buy_time: Vec<i64>,
    pub buy_price: Vec<f64>,
    pub sell_time: Vec<i64>,
    pub sell_price: Vec<f64>,
    pub qty: Vec<f64>,
    pub profit: Vec<f64>,
    pub exit_reason: Vec<String>,
    pub commission_buy: Vec<f64>,
    pub commission_sell: Vec<f64>,
}
// Métodos de TradeLog
impl TradeLog {
    pub fn new() -> Self {
        TradeLog {
            symbol: Vec::new(),
            buy_time: Vec::new(),
            buy_price: Vec::new(),
            sell_time: Vec::new(),
            sell_price: Vec::new(),
            qty: Vec::new(),
            profit: Vec::new(),
            exit_reason: Vec::new(),
            commission_buy: Vec::new(),
            commission_sell: Vec::new(),
        }
    }

    pub fn add(&mut self, symbol: String, buy_time: i64, buy_price: f64, sell_time: i64,
               sell_price: f64, qty: f64, profit: f64, exit_reason: String,
               commission_buy: f64, commission_sell: f64) {
        self.symbol.push(symbol);
        self.buy_time.push(buy_time);
        self.buy_price.push(buy_price);
        self.sell_time.push(sell_time);
        self.sell_price.push(sell_price);
        self.qty.push(qty);
        self.profit.push(profit);
        self.exit_reason.push(exit_reason);
        self.commission_buy.push(commission_buy);
        self.commission_sell.push(commission_sell);
    }
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolResult {
    pub trades: Vec<f64>,
    pub final_balance: f64,
    pub num_signals: usize,
    pub proportion_winners: f64,
    pub max_dd: f64,
    pub sharpe: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioResult {
    pub trades: Vec<f64>,
    pub final_balance: f64,
    pub num_signals: usize,
    pub proportion_winners: f64,
    pub max_dd: f64,
    pub sim_balance_history: SimBalance,
    pub trade_log: TradeLog,
    pub sharpe: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestResults {
    pub symbols: HashMap<String, SymbolResult>,
    pub portfolio: PortfolioResult,
}



#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimBalance {
    pub timestamp: Vec<i64>,
    pub balance: Vec<f64>,
}

// Métodos de SimBalance
impl SimBalance {
    pub fn new() -> Self {
        SimBalance {
            timestamp: Vec::new(),
            balance: Vec::new(),
        }
    }
}


// ============================
// Función principal
// ============================

pub fn run_grid_backtest(
    ohlcv_arrays: HashMap<String, OhlcvData>,
    sell_after: usize,
    initial_balance: f64,
    order_amount: f64,
    tp_pct: f64,
    sl_pct: f64,
    comi_pct: f64,
) -> BacktestResults {
    let comi_factor = comi_pct / 100.0;
    let mut cash = initial_balance;
    let mut num_signals_executed = 0;

    // Preparar sym_data
    let symbols: Vec<String> = ohlcv_arrays.keys().cloned().collect();
    let mut sym_data: HashMap<String, SymbolData> = HashMap::new();

    for sym in &symbols {
        let data = &ohlcv_arrays[sym];
        sym_data.insert(sym.clone(), SymbolData {
            ts: data.ts.clone(),
            close: data.close.clone(),
            high: data.high.clone(),
            low: data.low.clone(),
            signal: data.signal.clone(),
            len: data.ts.len(),
        });
    }

    // Señales por timestamp
    let mut signals_by_time: HashMap<i64, Vec<(String, usize)>> = HashMap::new();
    for sym in &symbols {
        let d = &sym_data[sym];
        for (idx, &is_signal) in d.signal.iter().enumerate() {
            if is_signal {
                let t_int = d.ts[idx];
                signals_by_time.entry(t_int)
                    .or_insert_with(Vec::new)
                    .push((sym.clone(), idx));
            }
        }
    }

    // Array ordenado de todos los timestamps
    let mut all_ts_set: std::collections::HashSet<i64> = std::collections::HashSet::new();
    for d in sym_data.values() {
        for &t in &d.ts {
            all_ts_set.insert(t);
        }
    }
    let mut all_timestamps_int: Vec<i64> = all_ts_set.into_iter().collect();
    all_timestamps_int.sort_unstable();

    // Estructuras auxiliares
    let mut trades: HashMap<String, Vec<f64>> = HashMap::new();
    let mut trade_times: HashMap<String, Vec<i64>> = HashMap::new();
    for sym in &symbols {
        trades.insert(sym.clone(), Vec::new());
        trade_times.insert(sym.clone(), Vec::new());
    }

    let mut trade_log = TradeLog::new();
    let mut sim_balance_cols = SimBalance::new();

    let mut open_positions_heap: BinaryHeap<HeapItem> = BinaryHeap::new();
    let mut counter = 0;

    let symbol_order: HashMap<String, usize> = symbols.iter()
        .enumerate()
        .map(|(i, s)| (s.clone(), i))
        .collect();

    let ts_index_map_by_sym: HashMap<String, HashMap<i64, usize>> = sym_data.iter()
        .map(|(sym, d)| {
            let map = d.ts.iter().enumerate()
                .map(|(idx, &t)| (t, idx))
                .collect();
            (sym.clone(), map)
        })
        .collect();

    // Función auxiliar para cerrar posición
    let close_position = |pos: &Position, exec_time: i64, exec_price: f64, 
                          exit_reason: String, cash: &mut f64,
                          trades: &mut HashMap<String, Vec<f64>>,
                          trade_times: &mut HashMap<String, Vec<i64>>,
                          trade_log: &mut TradeLog| {
        let qty = pos.qty;
        let buy_price = pos.buy_price;
        let commission_buy = pos.commission_buy;
        let commission_sell = if comi_factor != 0.0 {
            (qty * exec_price) * comi_factor
        } else {
            0.0
        };
        *cash += qty * exec_price - commission_sell;
        let profit = (exec_price - buy_price) * qty - commission_buy - commission_sell;

        trades.get_mut(&pos.symbol).unwrap().push(profit);
        trade_times.get_mut(&pos.symbol).unwrap().push(exec_time);

        trade_log.add(
            pos.symbol.clone(),
            pos.buy_time,
            buy_price,
            exec_time,
            exec_price,
            qty,
            profit,
            exit_reason,
            commission_buy,
            commission_sell,
        );
    };

    // Función para obtener precio
    let get_price_at_int = |sym: &str, t: i64, 
                             sym_data: &HashMap<String, SymbolData>,
                             ts_index_map: &HashMap<String, HashMap<i64, usize>>| -> Option<f64> {
        let d = &sym_data[sym];
        let idx_map = &ts_index_map[sym];
        if let Some(&idx) = idx_map.get(&t) {
            return Some(d.close[idx]);
        }
        // fallback: búsqueda binaria
        match d.ts.binary_search(&t) {
            Ok(idx) => Some(d.close[idx]),
            Err(idx) => {
                if idx > 0 {
                    Some(d.close[idx - 1])
                } else {
                    None
                }
            }
        }
    };

    // ============================
    // Bucle principal
    // ============================
    for &t_int in &all_timestamps_int {
        // Cerrar posiciones vencidas
        while let Some(item) = open_positions_heap.peek() {
            if item.time > t_int {
                break;
            }
            let mut item = open_positions_heap.pop().unwrap();
            if item.position.closed {
                continue;
            }

            if let Some(exec_price) = item.position.exec_price {
                if let Some(exec_time_int) = item.position.exec_time_int {
                    if exec_time_int <= t_int {
                        let exec_time = item.position.exec_time.unwrap();
                        let exit_reason = item.position.exit_reason.clone().unwrap();
                        close_position(&item.position, exec_time, exec_price, exit_reason,
                                       &mut cash, &mut trades, &mut trade_times, &mut trade_log);
                        item.position.closed = true;
                        continue;
                    }
                }
            }

            let sym = &item.position.symbol;
            let d = &sym_data[sym];
            let sell_ts_int = item.position.sell_time_int;
            let exec_price_opt = get_price_at_int(sym, sell_ts_int, &sym_data, &ts_index_map_by_sym);
            
            if let Some(exec_price) = exec_price_opt {
                close_position(&item.position, sell_ts_int, exec_price, "SELL_AFTER".to_string(),
                               &mut cash, &mut trades, &mut trade_times, &mut trade_log);
            } else {
                let exec_price = d.close[d.len - 1];
                let last_time = d.ts[d.len - 1];
                close_position(&item.position, last_time, exec_price, "FORCED_LAST".to_string(),
                               &mut cash, &mut trades, &mut trade_times, &mut trade_log);
            }
            item.position.closed = true;
        }

        // Actualizar balance si hay posiciones abiertas
        if !open_positions_heap.is_empty() {
            let positions_value: f64 = open_positions_heap.iter()
                .filter(|item| !item.position.closed)
                .map(|item| {
                    let price = get_price_at_int(&item.position.symbol, t_int, &sym_data, &ts_index_map_by_sym)
                        .unwrap_or(0.0);
                    item.position.qty * price
                })
                .sum();
            
            sim_balance_cols.timestamp.push(t_int);
            sim_balance_cols.balance.push(cash + positions_value);
            continue;
        }

        // Abrir nuevas posiciones
        if let Some(mut events) = signals_by_time.get(&t_int).cloned() {
            events.sort_by_key(|(sym, _)| symbol_order[sym]);
            
            for (sym, buy_idx) in events {
                if cash < order_amount {
                    break;
                }

                let d = &sym_data[&sym];
                let price_t = d.close[buy_idx];
                let qty = order_amount / price_t;
                let commission_buy = if comi_factor != 0.0 {
                    order_amount * comi_factor
                } else {
                    0.0
                };
                cash -= order_amount + commission_buy;
                num_signals_executed += 1;

                let sell_idx = (buy_idx + sell_after).min(d.len - 1);
                let sell_time_dt = d.ts[sell_idx];
                let sell_time_int = d.ts[sell_idx];
                
                let tp_price = if tp_pct != 0.0 {
                    price_t * (1.0 + tp_pct / 100.0)
                } else {
                    f64::INFINITY
                };
                
                let sl_price = if sl_pct != 0.0 {
                    price_t * (1.0 - sl_pct / 100.0)
                } else {
                    f64::NEG_INFINITY
                };

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

                // Detección intravela
                let mut intravela_detected = false;
                if tp_price.is_finite() || sl_price.is_finite() {
                    if let (Some(high_arr), Some(low_arr)) = (&d.high, &d.low) {
                        let start = buy_idx + 1;
                        let end = sell_idx;
                        if end >= start {
                            let mut tp_first: Option<usize> = None;
                            let mut sl_first: Option<usize> = None;

                            for i in start..=end {
                                if tp_price.is_finite() && high_arr[i] >= tp_price && tp_first.is_none() {
                                    tp_first = Some(i);
                                }
                                if sl_price.is_finite() && low_arr[i] <= sl_price && sl_first.is_none() {
                                    sl_first = Some(i);
                                }
                                if tp_first.is_some() && sl_first.is_some() {
                                    break;
                                }
                            }

                            let (chosen_idx, exit_reason, exec_price) = match (tp_first, sl_first) {
                                (Some(tp_idx), Some(sl_idx)) => {
                                    if sl_idx <= tp_idx {
                                        (sl_idx, "SL".to_string(), sl_price)
                                    } else {
                                        (tp_idx, "TP".to_string(), tp_price)
                                    }
                                },
                                (None, Some(sl_idx)) => (sl_idx, "SL".to_string(), sl_price),
                                (Some(tp_idx), None) => (tp_idx, "TP".to_string(), tp_price),
                                (None, None) => (0, String::new(), 0.0), // no se usa
                            };

                            if tp_first.is_some() || sl_first.is_some() {
                                let exec_time_dt = d.ts[chosen_idx];
                                let exec_time_int = d.ts[chosen_idx];
                                position.exec_price = Some(exec_price);
                                position.exec_time = Some(exec_time_dt);
                                position.exec_time_int = Some(exec_time_int);
                                position.exit_reason = Some(exit_reason);
                                intravela_detected = true;
                            }
                        }
                    }
                }

                if intravela_detected {
                    let exec_time_int = position.exec_time_int.unwrap();
                    open_positions_heap.push(HeapItem {
                        time: exec_time_int,
                        counter,
                        position,
                    });
                    counter += 1;
                } else {
                    open_positions_heap.push(HeapItem {
                        time: sell_time_int,
                        counter,
                        position,
                    });
                    counter += 1;
                }
            }
        }

        // Registrar balance actual
        let positions_value: f64 = open_positions_heap.iter()
            .filter(|item| !item.position.closed)
            .map(|item| {
                let price = get_price_at_int(&item.position.symbol, t_int, &sym_data, &ts_index_map_by_sym)
                    .unwrap_or(0.0);
                item.position.qty * price
            })
            .sum();
        
        sim_balance_cols.timestamp.push(t_int);
        sim_balance_cols.balance.push(cash + positions_value);
    }

    // ============================
    // Cierre final de posiciones
    // ============================
    while let Some(item) = open_positions_heap.pop() {
        if item.position.closed {
            continue;
        }

        let sym = &item.position.symbol;
        let d = &sym_data[sym];

        if let Some(exec_price) = item.position.exec_price {
            let exec_time = item.position.exec_time.unwrap();
            let exit_reason = item.position.exit_reason.clone().unwrap();
            close_position(&item.position, exec_time, exec_price, exit_reason,
                           &mut cash, &mut trades, &mut trade_times, &mut trade_log);
        } else {
            let sell_ts_int = item.position.sell_time_int;
            let exec_price_opt = get_price_at_int(sym, sell_ts_int, &sym_data, &ts_index_map_by_sym);
            
            if let Some(exec_price) = exec_price_opt {
                close_position(&item.position, sell_ts_int, exec_price, "SELL_AFTER".to_string(),
                               &mut cash, &mut trades, &mut trade_times, &mut trade_log);
            } else {
                let exec_price = d.close[d.len - 1];
                let last_time = d.ts[d.len - 1];
                close_position(&item.position, last_time, exec_price, "FORCED_LAST".to_string(),
                               &mut cash, &mut trades, &mut trade_times, &mut trade_log);
            }
        }
    }

    // ============================
    // Resultados finales
    // ============================
    let final_balance = cash;
    let mut all_trades: Vec<f64> = Vec::new();
    for sym in &symbols {
        all_trades.extend(&trades[sym]);
    }
    let num_trades = all_trades.len();
    let proportion_winners = if num_trades > 0 {
        all_trades.iter().filter(|&&t| t > 0.0).count() as f64 / num_trades as f64
    } else {
        f64::NAN
    };

    // Índice de timestamps
    let ts_index_map: HashMap<i64, usize> = all_timestamps_int.iter()
        .enumerate()
        .map(|(i, &t)| (t, i))
        .collect();

    let mut max_dd_by_symbol: HashMap<String, f64> = HashMap::new();
    let mut final_balance_by_symbol: HashMap<String, f64> = HashMap::new();
    let mut equity_by_symbol: HashMap<String, Vec<f64>> = HashMap::new();

    for sym in &symbols {
        let mut profits_series = vec![0.0; all_timestamps_int.len()];
        for (profit, t_close) in trades[sym].iter().zip(&trade_times[sym]) {
            if let Some(&idx) = ts_index_map.get(t_close) {
                profits_series[idx] += profit;
            } else {
                let idx = match all_timestamps_int.binary_search(t_close) {
                    Ok(i) => i,
                    Err(i) => if i > 0 { i - 1 } else { continue },
                };
                if idx < profits_series.len() {
                    profits_series[idx] += profit;
                }
            }
        }

        let mut equity = vec![initial_balance];
        for p in profits_series {
            equity.push(equity.last().unwrap() + p);
        }
        equity.remove(0);

        let (max_dd, final_bal) = if !equity.is_empty() {
            let mut cummax = equity[0];
            let mut max_drawdown = 0.0;
            for &eq in &equity {
                if eq > cummax {
                    cummax = eq;
                }
                let dd = if cummax == 0.0 { 0.0 } else { (cummax - eq) / cummax };
                if dd > max_drawdown {
                    max_drawdown = dd;
                }
            }
            (max_drawdown, equity[equity.len() - 1])
        } else {
            (0.0, initial_balance)
        };

        equity_by_symbol.insert(sym.clone(), equity);
        max_dd_by_symbol.insert(sym.clone(), max_dd);
        final_balance_by_symbol.insert(sym.clone(), final_bal);
    }

    // Max DD del portfolio
    let sim_values = if sim_balance_cols.balance.is_empty() {
        vec![initial_balance]
    } else {
        sim_balance_cols.balance.clone()
    };

    let max_dd_portfolio = if !sim_values.is_empty() {
        let mut cummax = sim_values[0];
        let mut max_drawdown = 0.0;
        for &val in &sim_values {
            if val > cummax {
                cummax = val;
            }
            let dd = if cummax == 0.0 { 0.0 } else { (cummax - val) / cummax };
            if dd > max_drawdown {
                max_drawdown = dd;
            }
        }
        max_drawdown
    } else {
        0.0
    };

    // ============================
    // Cálculo del Sharpe
    // ============================
    let compute_annualized_sharpe = |equity_arr: &[f64], time_index: &[i64]| -> f64 {
        if equity_arr.len() < 2 {
            return f64::NAN;
        }

        let mut returns = Vec::new();
        for i in 1..equity_arr.len() {
            if equity_arr[i - 1] != 0.0 {
                let ret = (equity_arr[i] / equity_arr[i - 1]) - 1.0;
                if ret.is_finite() {
                    returns.push(ret);
                }
            }
        }

        if returns.is_empty() {
            return f64::NAN;
        }

        let median_delta_s = if time_index.len() >= 2 {
            let mut deltas: Vec<f64> = Vec::new();
            for i in 1..time_index.len() {
                let delta_ns = (time_index[i] - time_index[i - 1]) as f64;
                let delta_s = delta_ns / 1e9;
                if delta_s > 0.0 {
                    deltas.push(delta_s);
                }
            }
            if deltas.is_empty() {
                24.0 * 3600.0
            } else {
                deltas.sort_by(|a, b| a.partial_cmp(b).unwrap());
                deltas[deltas.len() / 2]
            }
        } else {
            24.0 * 3600.0
        };

        let periods_per_year = if median_delta_s > 0.0 {
            (365.0 * 24.0 * 3600.0) / median_delta_s
        } else {
            252.0
        };

        let mean_periodic: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance: f64 = returns.iter()
            .map(|&r| (r - mean_periodic).powi(2))
            .sum::<f64>() / returns.len() as f64;
        let std_periodic = variance.sqrt();

        if !std_periodic.is_finite() || std_periodic == 0.0 {
            return f64::NAN;
        }

        let annualized_mean = mean_periodic * periods_per_year;
        let annualized_std = std_periodic * periods_per_year.sqrt();

        if annualized_std == 0.0 {
            f64::NAN
        } else {
            annualized_mean / annualized_std
        }
    };

    // Sharpe por símbolo
    let mut sharpe_by_symbol: HashMap<String, f64> = HashMap::new();
    for sym in &symbols {
        let equity = &equity_by_symbol[sym];
        let sharpe = compute_annualized_sharpe(equity, &all_timestamps_int);
        sharpe_by_symbol.insert(sym.clone(), sharpe);
    }

    // Sharpe del portfolio
    let sim_ts_arr = if !sim_balance_cols.timestamp.is_empty() {
        &sim_balance_cols.timestamp
    } else {
        &all_timestamps_int
    };
    let sharpe_portfolio = compute_annualized_sharpe(&sim_values, sim_ts_arr);

    // ============================
    // Construcción de resultados
    // ============================
    let mut symbol_results: HashMap<String, SymbolResult> = HashMap::new();
    for sym in &symbols {
        let sym_trades = &trades[sym];
        let prop_winners = if sym_trades.is_empty() {
            f64::NAN
        } else {
            sym_trades.iter().filter(|&&t| t > 0.0).count() as f64 / sym_trades.len() as f64
        };

        symbol_results.insert(sym.clone(), SymbolResult {
            trades: sym_trades.clone(),
            final_balance: final_balance_by_symbol[sym],
            num_signals: sym_trades.len(),
            proportion_winners: prop_winners,
            max_dd: max_dd_by_symbol[sym],
            sharpe: sharpe_by_symbol[sym],
        });
    }

    let portfolio = PortfolioResult {
        trades: all_trades,
        final_balance,
        num_signals: num_signals_executed,
        proportion_winners,
        max_dd: max_dd_portfolio,
        sim_balance_history: sim_balance_cols,
        trade_log,
        sharpe: sharpe_portfolio,
    };

    BacktestResults {
        symbols: symbol_results,
        portfolio,
    }
}

