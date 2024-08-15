use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

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
