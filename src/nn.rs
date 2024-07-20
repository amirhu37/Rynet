use derive_more::Display;
use pyo3::{
    class,
    exceptions::PyZeroDivisionError,
    prelude::*,
    types::{PyCode, PyDict, PyList, PyString, PyTuple, PyType},
    PyClass, PyErrArguments,
};
use std::{borrow::Borrow, collections::HashMap};

use crate::{_py_run, MODULE};
use pyo3::Bound;

#[allow(unconditional_recursion)]
#[derive(
    Debug,
    //  Display,
    Clone,
)]
#[pyclass(
module = "nn", 
// name = "44" , 
unsendable,
get_all,
set_all,
subclass,
sequence,
dict ,
)]
// #[pyo3(text_signature = "$cls(*args , **kwargs)" )]
// #[display(fmt = "")]
pub struct Mlp {
    pub args: Option<Py<PyTuple>>,
    pub kwargs: Option<Py<PyDict>>,
}

#[pymethods]
impl Mlp {
    #[new]
    #[pyo3(signature = (args = None, kwargs = None) ,)]
    pub fn __new__(py: Python, args: Option<&PyTuple>, kwargs: Option<&PyDict>) -> Self {
        Mlp {
            args: args.filter(|d| !d.is_empty()).map(|d| d.into_py(py)),
            kwargs: kwargs.filter(|d| !d.is_empty()).map(|d| d.into_py(py)),
        }
    }

    fn parameters(slf: &Bound<Self>, py: Python) -> Py<PyDict> {
        // acces dict of the class
        let cx = _py_run(&slf.clone().unbind().as_any(), "value.__dict__").unwrap();
        let values = cx.call_method0(py, "values").unwrap();
        // todo!()
        let v: &Bound<PyAny> = values.downcast_bound::<PyAny>(py).unwrap();
        // println!("v r {v}");
        cx
    }

    pub fn forward(&self, x: PyObject) -> PyResult<PyObject> {
        // TODO
        return Ok(x);
    }

    fn __str__(slf: &Bound<Self>) -> PyResult<String> {
        let class_name: String = slf.get_type().qualname()?;

        Ok(format!("{}", class_name))
    }

    fn __repr__(slf: &Bound<Self>) -> PyResult<String> {
        // This is the equivalent of `self.__class__.__name__` in Python.
        let class_name = slf.get_type().qualname()?;
        Ok(format!("{}", class_name))
    }
}

// impl<'source> FromPyObject<'source> for Mlp {
//     fn extract(obj: &PyAny) -> PyResult<Self> {
//         let args = obj.getattr("args")?;
//         let kwargs = obj.getattr("kwargs")?;
//         Ok(Mlp {
//             args: args.extract()?,
//             kwargs: kwargs.extract()?,
//             })
//             }
//             }
