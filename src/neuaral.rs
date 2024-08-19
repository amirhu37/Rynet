use pyo3::types::{PyDict, PyTuple};
use pyo3::prelude::*;

use pyo3::Bound;


#[allow(unconditional_recursion)]
#[derive(
    Debug,
    //  Display,
    Clone,
)]
#[pyclass(module = "nn", unsendable, get_all, set_all, subclass, sequence, dict)]
// #[pyo3(text_signature = "$cls(*args , **kwargs)" )]
// #[display(fmt = "")]
pub struct Neuaral {}
#[pymethods]
impl Neuaral {
    #[new]
    #[pyo3(signature = (*args , **kwargs ) ,)]
    #[allow(unused_variables)]
    pub fn __new__(py: Python, args: &Bound<PyTuple>, kwargs: Option<&Bound<PyDict>>) -> Self {
        Neuaral {  } }

    fn parameters<'py>(slf: &Bound<Self>, _py: Python<'py>) -> Py<PyDict> {
        let dict = slf
            .getattr("__dict__")
            .unwrap()
            .downcast::<PyDict>()
            .unwrap()
            .clone() ;
        let _binding = dict.as_gil_ref().downcast::<PyDict>().unwrap();
        return dict.unbind();
    }

    pub fn forward(&self, x: PyObject) -> PyResult<PyObject> {
        // TODO
        return Ok(x);
    }
    pub fn __call__(&self, x: PyObject) -> PyObject {
        self.forward(x).expect("call error")
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


// #[pymodule]
// #[pyo3(name = "neuaral")]
// pub fn neuralmodule(_py: Python, m: &Bound<PyModule>) -> PyResult<()>{
//     add_class!(m, Neuaral);

//     Ok(())
// }