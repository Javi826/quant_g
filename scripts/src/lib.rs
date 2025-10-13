use pyo3::prelude::*;
use std::collections::HashMap;
use serde_json;

// Importa tu módulo con la lógica Rust
mod backtest;
use backtest::{run_grid_backtest, OhlcvData};

// Clase Python
#[pyclass]
#[derive(Clone)]
pub struct PyOhlcvData {
    #[pyo3(get, set)]
    pub ts: Vec<i64>,
    #[pyo3(get, set)]
    pub close: Vec<f64>,
    #[pyo3(get, set)]
    pub high: Option<Vec<f64>>,
    #[pyo3(get, set)]
    pub low: Option<Vec<f64>>,
    #[pyo3(get, set)]
    pub signal: Vec<bool>,
}

#[pymethods]
impl PyOhlcvData {
    #[new]
    #[pyo3(signature = (ts, close, signal=None, high=None, low=None))]
    pub fn new(
        ts: Vec<i64>,
        close: Vec<f64>,
        signal: Option<Vec<bool>>,
        high: Option<Vec<f64>>,
        low: Option<Vec<f64>>,
    ) -> Self {
        let len = ts.len(); // guarda la longitud antes de mover
        PyOhlcvData {
            ts,
            close,
            high,
            low,
            signal: signal.unwrap_or_else(|| vec![false; len]),
        }
    }
}

// Función Python
#[pyfunction]
fn run_grid_backtest_py(
    ohlcv_py: HashMap<String, PyOhlcvData>,
    sell_after: usize,
    initial_balance: f64,
    order_amount: f64,
    tp_pct: f64,
    sl_pct: f64,
    comi_pct: f64,
) -> PyResult<String> {
    let mut rust_data: HashMap<String, OhlcvData> = HashMap::new();
    for (sym, d) in ohlcv_py {
        rust_data.insert(
            sym,
            OhlcvData {
                ts: d.ts,
                close: d.close,
                high: d.high,
                low: d.low,
                signal: d.signal,
            },
        );
    }

    let results = run_grid_backtest(
        rust_data, sell_after, initial_balance, order_amount, tp_pct, sl_pct, comi_pct
    );

    Ok(serde_json::to_string(&results).unwrap())
}

// Módulo Python
#[pymodule]
fn rust_backtest(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyOhlcvData>()?;
    m.add_function(wrap_pyfunction!(run_grid_backtest_py, m)?)?;
    Ok(())
}
