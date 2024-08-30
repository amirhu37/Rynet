use ndarray::{Array, Array1};
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyDict, PyList};
use rand::Rng;
use crate::{ArrayAs, OneDim, TwoDim};

pub fn extract_keys(dict: &PyDict) -> PyResult<Vec<String>> {
    // Use the keys() method to get a PyList of keys
    let keys: &PyList = dict.keys();

    // Convert the PyList to a Vec<String>
    let keys_vec: Vec<String> = keys
        .iter()
        .map(|key| key.to_string())
        .collect::<Vec<String>>();

    Ok(keys_vec)
}

pub fn extract_values(dict: &PyDict) -> PyResult<&PyList> {
    // Use the keys() method to get a PyList of keys
    let values: &PyList = dict.values();

    // // Convert the PyList to a Vec<String>
    // let keys_vec: Vec<PyAny> = values
    //     .iter()
    //     .map(|key| key.to_string())
    //     .collect::<Vec<PyOb>>();

    Ok(values)
}



pub fn random_weight(n: usize, m: usize) -> PyResult<ArrayAs<f32, TwoDim>> {
    let mut rng = rand::thread_rng();
    let mut array: ArrayAs<f32, TwoDim> = Array::zeros((n, m));

    for i in 0..n {
        for j in 0..m {
            array[[i, j]] = rng.gen::<f32>();
        }
    }
    Ok(array)
}

pub fn stack(weight : ArrayAs<f32, TwoDim>, bias : ArrayAs<f32, OneDim> )->  PyResult<ArrayAs<f32, TwoDim>> {
    // stack both array into one
    let mut stacked_array = Array::zeros((weight.shape()[0], weight.shape()[1]
    + bias.shape()[0]));
    for i in 0..weight.shape()[0] {
        for j in 0..weight.shape()[1] {
            stacked_array[[i, j]] = weight[[i, j]];
            }
            for j in 0..bias.shape()[0] {
                stacked_array[[i, weight.shape()[1] + j]] = bias[[j]];
                }
        }
        Ok(stacked_array)

}


pub fn random_bias<'py>(n: usize) -> PyResult<ArrayAs<f32, OneDim>> {
    let mut rng = rand::thread_rng();

    let mut array: ArrayAs<f32, OneDim> = Array1::zeros(n);
    for i in 0..n {
        array[i] = rng.gen::<f32>();
    }
    Ok(array)
}

pub fn _py_run(value: &Bound<PyAny>, command: &str) -> PyResult<Py<PyDict>> {
    Python::with_gil(|py| {
        let locals = [("value", value)].into_py_dict_bound(py);
        let result = py.eval_bound(command, None, Some(&locals))?.unbind();
        let py_dict = result.downcast_bound(py).unwrap().clone().unbind();
        Ok(py_dict.into())
    })
}

#[macro_export]
macro_rules! add_class {
    ($module : ident , $($class : ty), +) => {
        $(
            $module.add_class::<$class>()?;
        )+

    };
}
#[allow(unused_macros)]
macro_rules! add_function {
    ($module : ident , $($function : ident), +) => {
        $(
           $module.add_wrapped(wrap_pyfunction!($function))?;
        )+
    };
}
