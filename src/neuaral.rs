use pyo3::{
    prelude::*,
    types::{IntoPyDict, PyDict, PyTuple},
};

use pyo3::Bound;

#[allow(unconditional_recursion)]
#[derive(
    Debug,
    //  Display,
    Clone,
)]
#[pyclass(
module = "nn", 
unsendable,
get_all,
set_all,
subclass,
sequence,
dict ,
)]
// #[pyo3(text_signature = "$cls(*args , **kwargs)" )]
// #[display(fmt = "")]
pub struct Neuaral {
    pub args: Option<Py<PyTuple>>,
    pub kwargs: Option<Py<PyDict>>,
}

#[pymethods]
impl Neuaral {
    #[new]
    #[pyo3(signature = (args = None, kwargs = None) ,)]
    pub fn __new__(
        py: Python,
        args: Option<&Bound<PyTuple>>,
        kwargs: Option<&Bound<PyDict>>,
    ) -> Self {
        let kw: Option<Py<PyDict>> = kwargs
            .filter(|d| !d.is_empty())
            .map(|d| d.into_py_dict_bound(py).unbind());
        let arg: Option<Py<PyTuple>> = args.filter(|d| !d.is_empty()).map(|d| d.into_py(py));
        Neuaral {
            args: match arg {
                Some(arg) => Some(arg),
                None => None,
            },
            kwargs: match kw {
                Some(kw) => Some(kw),
                None => None,
            },
        }
    }

    fn parameters<'py>(slf: &Bound<Self>, _py: Python<'py>) -> Py<PyDict> {
        // acces dict of the class
        let dict = slf
            .getattr("__dict__")
            .unwrap()
            .downcast::<PyDict>()
            .unwrap()
            .clone()
            // .unbind()
            ;
        let _binding = dict.as_gil_ref().downcast::<PyDict>().unwrap();
        // let v = binding.values();
        // let v1 =  &v[0];
        // println!("v1 : {0}", v1.weights);
        return dict.unbind();
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
