use std::{collections::HashMap, fmt::format};

use pyo3::{prelude::*, types::PyDict};

use crate::Ndarray;


#[pyclass(name = "mlp", unsendable, subclass, sequence, dict,)]
#[pyo3(text_signature = "()")]
pub struct mlp;


#[pymethods]
impl mlp {
    #[new]
    pub fn new() -> PyResult<Self> {
        Ok(Self{})
    }

    pub fn forward(&mut self, x: PyObject) -> PyResult<PyObject> {
        // TODO
        return Ok(x);
    }

    fn __str__(&self) -> PyResult<String>{
        Ok(
            format!("nn network" )
        )
    }

    fn __repr__(&self)-> PyResult<String>{
        Ok(format!("__repr__"))
    }
    #[getter]
    fn __dict__(&self)-> PyResult<PyObject>{
        let __dict__: HashMap<String, PyObject> = HashMap::new();
        Python::with_gil(|py|
        {Ok(__dict__.to_object(py))})

    }
}
