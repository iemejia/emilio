//! PyO3 bindings: expose emilio to Python as a native module.
//!
//! The module name is `emilio` — import as:
//!   import emilio
//!   emilio.eml_matmul(a, b)

use numpy::ndarray::ArrayD;
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::prelude::*;

use crate::eml_ops;

// ─── Scalar ops (take and return plain f64) ─────────────────────────────────

#[pyfunction]
fn eml(x: f64, y: f64) -> f64 {
    eml_ops::to_r(eml_ops::eml(eml_ops::to_c(x), eml_ops::to_c(y)))
}

#[pyfunction]
fn eml_exp(x: f64) -> f64 {
    eml_ops::to_r(eml_ops::eml_exp(eml_ops::to_c(x)))
}

#[pyfunction]
fn eml_ln(x: f64) -> f64 {
    eml_ops::to_r(eml_ops::eml_ln(eml_ops::to_c(x)))
}

#[pyfunction]
fn eml_sub(a: f64, b: f64) -> f64 {
    eml_ops::to_r(eml_ops::eml_sub(eml_ops::to_c(a), eml_ops::to_c(b)))
}

#[pyfunction]
fn eml_neg(x: f64) -> f64 {
    eml_ops::to_r(eml_ops::eml_neg(eml_ops::to_c(x)))
}

#[pyfunction]
fn eml_add(a: f64, b: f64) -> f64 {
    eml_ops::to_r(eml_ops::eml_add(eml_ops::to_c(a), eml_ops::to_c(b)))
}

#[pyfunction]
fn eml_inv(x: f64) -> f64 {
    eml_ops::to_r(eml_ops::eml_inv(eml_ops::to_c(x)))
}

#[pyfunction]
fn eml_mul(a: f64, b: f64) -> f64 {
    eml_ops::to_r(eml_ops::eml_mul(eml_ops::to_c(a), eml_ops::to_c(b)))
}

#[pyfunction]
fn eml_div(a: f64, b: f64) -> f64 {
    eml_ops::to_r(eml_ops::eml_div(eml_ops::to_c(a), eml_ops::to_c(b)))
}

#[pyfunction]
fn eml_pow(a: f64, b: f64) -> f64 {
    eml_ops::to_r(eml_ops::eml_pow(eml_ops::to_c(a), eml_ops::to_c(b)))
}

#[pyfunction]
fn eml_sqrt(x: f64) -> f64 {
    eml_ops::to_r(eml_ops::eml_sqrt(eml_ops::to_c(x)))
}

#[pyfunction]
fn eml_gelu(x: f64) -> f64 {
    eml_ops::to_r(eml_ops::eml_gelu(eml_ops::to_c(x)))
}

// ─── Array ops (numpy in, numpy out) ────────────────────────────────────────

#[pyfunction]
fn eml_softmax<'py>(py: Python<'py>, x: PyReadonlyArrayDyn<'py, f64>) -> Bound<'py, PyArrayDyn<f64>> {
    let x_slice = x.as_slice().expect("contiguous array required");
    let result = eml_ops::eml_softmax(x_slice);
    let shape: Vec<usize> = x.shape().to_vec();
    ArrayD::from_shape_vec(shape, result)
        .expect("shape mismatch")
        .into_pyarray(py)
}

#[pyfunction]
fn eml_layer_norm<'py>(
    py: Python<'py>,
    x: PyReadonlyArrayDyn<'py, f64>,
    gamma: PyReadonlyArrayDyn<'py, f64>,
    beta: PyReadonlyArrayDyn<'py, f64>,
    eps: f64,
) -> Bound<'py, PyArrayDyn<f64>> {
    let shape: Vec<usize> = x.shape().to_vec();
    let cols = *shape.last().expect("empty shape");
    let total: usize = shape.iter().product();
    let rows = total / cols;
    let result = eml_ops::eml_layer_norm(
        x.as_slice().expect("contiguous"),
        gamma.as_slice().expect("contiguous"),
        beta.as_slice().expect("contiguous"),
        rows,
        cols,
        eps,
    );
    ArrayD::from_shape_vec(shape, result)
        .expect("shape mismatch")
        .into_pyarray(py)
}

#[pyfunction]
fn eml_matmul<'py>(
    py: Python<'py>,
    a: PyReadonlyArrayDyn<'py, f64>,
    b: PyReadonlyArrayDyn<'py, f64>,
) -> Bound<'py, PyArrayDyn<f64>> {
    let a_shape = a.shape();
    let b_shape = b.shape();
    let rows = a_shape[0];
    let inner = a_shape[1];
    let cols = b_shape[1];
    let result = eml_ops::eml_matmul(
        a.as_slice().expect("contiguous"),
        b.as_slice().expect("contiguous"),
        rows,
        inner,
        cols,
    );
    ArrayD::from_shape_vec(vec![rows, cols], result)
        .expect("shape mismatch")
        .into_pyarray(py)
}

#[pyfunction]
fn eml_gelu_vec<'py>(py: Python<'py>, x: PyReadonlyArrayDyn<'py, f64>) -> Bound<'py, PyArrayDyn<f64>> {
    let result = eml_ops::eml_gelu_vec(x.as_slice().expect("contiguous"));
    let shape: Vec<usize> = x.shape().to_vec();
    ArrayD::from_shape_vec(shape, result)
        .expect("shape mismatch")
        .into_pyarray(py)
}

#[pyfunction]
fn eml_add_vec<'py>(
    py: Python<'py>,
    a: PyReadonlyArrayDyn<'py, f64>,
    b: PyReadonlyArrayDyn<'py, f64>,
) -> Bound<'py, PyArrayDyn<f64>> {
    let result = eml_ops::eml_add_vec(
        a.as_slice().expect("contiguous"),
        b.as_slice().expect("contiguous"),
    );
    let shape: Vec<usize> = a.shape().to_vec();
    ArrayD::from_shape_vec(shape, result)
        .expect("shape mismatch")
        .into_pyarray(py)
}

#[pyfunction]
fn eml_mul_vec<'py>(
    py: Python<'py>,
    a: PyReadonlyArrayDyn<'py, f64>,
    b: PyReadonlyArrayDyn<'py, f64>,
) -> Bound<'py, PyArrayDyn<f64>> {
    let result = eml_ops::eml_mul_vec(
        a.as_slice().expect("contiguous"),
        b.as_slice().expect("contiguous"),
    );
    let shape: Vec<usize> = a.shape().to_vec();
    ArrayD::from_shape_vec(shape, result)
        .expect("shape mismatch")
        .into_pyarray(py)
}

// ─── Module definition ──────────────────────────────────────────────────────

#[pymodule]
fn emilio(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Scalars
    m.add_function(wrap_pyfunction!(eml, m)?)?;
    m.add_function(wrap_pyfunction!(eml_exp, m)?)?;
    m.add_function(wrap_pyfunction!(eml_ln, m)?)?;
    m.add_function(wrap_pyfunction!(eml_sub, m)?)?;
    m.add_function(wrap_pyfunction!(eml_neg, m)?)?;
    m.add_function(wrap_pyfunction!(eml_add, m)?)?;
    m.add_function(wrap_pyfunction!(eml_inv, m)?)?;
    m.add_function(wrap_pyfunction!(eml_mul, m)?)?;
    m.add_function(wrap_pyfunction!(eml_div, m)?)?;
    m.add_function(wrap_pyfunction!(eml_pow, m)?)?;
    m.add_function(wrap_pyfunction!(eml_sqrt, m)?)?;
    m.add_function(wrap_pyfunction!(eml_gelu, m)?)?;
    // Arrays
    m.add_function(wrap_pyfunction!(eml_softmax, m)?)?;
    m.add_function(wrap_pyfunction!(eml_layer_norm, m)?)?;
    m.add_function(wrap_pyfunction!(eml_matmul, m)?)?;
    m.add_function(wrap_pyfunction!(eml_gelu_vec, m)?)?;
    m.add_function(wrap_pyfunction!(eml_add_vec, m)?)?;
    m.add_function(wrap_pyfunction!(eml_mul_vec, m)?)?;
    Ok(())
}
