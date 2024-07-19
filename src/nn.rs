use std::{collections::HashMap, ops::Bound};

use pyo3::{prelude::*, types::{IntoPyDict, PyDict, PyTuple}};

#[pyclass(module = "nn", name = "mlp", unsendable, subclass, sequence, dict)]
#[pyo3(text_signature = "(*args , **kwargs)")]
pub struct Mlp{
    pub args : Option<Py<PyTuple>>,
    pub kwargs : Option<Py<PyDict>>,
}

#[pymethods]
impl Mlp {
    #[new]
    #[pyo3(signature = (args = None, kwargs = None))]
    pub fn __new__(py: Python, args: Option<&PyTuple>, kwargs: Option<&PyDict>) -> Self {
        Mlp {
            args: args.filter(|d| !d.is_empty())
                     .map(|d|  d.into_py(py)),
            kwargs: kwargs
                .filter(|d| !d.is_empty())
                .map(|d|  d.into_py(py)),
        }
    }

    pub fn forward(&mut self, x: PyObject) -> PyResult<PyObject> {
        // TODO
        return Ok(x);
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(format!("nn network"))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("__repr__"))
    }
    #[getter]
    fn __dict__(&self) -> PyResult<PyObject> {
        let __dict__: HashMap<String, PyObject> = HashMap::new();
        Python::with_gil(|py| Ok(__dict__.to_object(py)))
    }
}
