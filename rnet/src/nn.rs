use numpy::PyArrayDyn;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::Bound as PyBound;
use crate::add_class;
use crate::linear::Linear;
use crate::model::Model;

#[pymodule]
#[pyo3(name = "nn")]
pub fn register(module: &PyBound<PyModule> ) -> PyResult<()> {

    add_class!(module,Linear, Model);

    Ok(())
}

// pub trait Nn<T>{
//     fn __new__<'py>(
//         py: Python,
//         in_features: usize,
//         out_features: usize,
//         is_bias: Option<bool>,
//         trainable: Option<bool>,
//     ) -> PyResult<Self> where Self: Sized;
//     fn forward(slf: &PyBound<Self>, x: &PyBound<PyAny>, py: Python<'_>) -> PyResult<PyObject> where Self: Sized;
//     fn parameters<'py>(slf: &PyBound<Self>, _py: Python<'py>) -> PyResult<Py<PyDict>> where Self: Sized;
//     fn __call__(slf: &PyBound<Self>, py: Python<'_>, value: &PyBound<PyAny>) -> PyResult<PyObject> where Self: Sized;
// }