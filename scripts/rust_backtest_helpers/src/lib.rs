use pyo3::prelude::*;
use numpy::PyArray1;

/// get_price_at_int_fast — versión Rust de la función Python
#[pyfunction]
fn get_price_at_int_fast(ts_arr: &PyArray1<i64>, close_arr: &PyArray1<f64>, t_int: i64) -> Option<f64> {
    // Acceso a los datos dentro de bloques `unsafe`
    let ts = unsafe { ts_arr.as_slice().unwrap() };
    let close = unsafe { close_arr.as_slice().unwrap() };

    if ts.is_empty() {
        return None;
    }

    // Búsqueda binaria rápida
    let mut left = 0;
    let mut right = ts.len();

    while left < right {
        let mid = (left + right) / 2;
        if ts[mid] < t_int {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    if left == 0 {
        None
    } else {
        Some(close[left - 1])
    }
}

/// Módulo PyO3
#[pymodule]
fn rust_backtest_helpers(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_price_at_int_fast, m)?)?;
    Ok(())
}
