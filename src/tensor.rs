
use std::any::type_name_of_val;
use std::fmt::{Display, Formatter};


use numpy::npyffi::npy_float;
use numpy::{IntoPyArray, IxDyn, PyArray, PyArrayDyn};

use pyo3::exceptions::PyTypeError;
use pyo3::Bound as PyBound;
use pyo3::prelude::*;

// Define the TensorTrait trait
// pub trait TensorTrait {
//     // We can define some common behavior here if needed
// }

// // Implement TensorTrait for ArrayAs<npy_float, OneDim>
// impl TensorTrait for ArrayAs<npy_float, OneDim> {}

// // Implement TensorTrait for ArrayAs<npy_float, TwoDim>
// impl TensorTrait for ArrayAs<npy_float, TwoDim> {}

// // Implement TensorTrait for ArrayAs<npy_float, ThreeDim>
// impl TensorTrait for ArrayAs<npy_float, ThreeDim> {}
use pyo3::types::{PyAny, PyFloat, PyList};


#[allow(dead_code)]
trait TensorInput {}

impl TensorInput for Py<PyList> {}

impl TensorInput for Py<PyArrayDyn<npy_float>> {}

#[pyclass(name = "Tensor", unsendable, subclass, sequence, frozen, dict)]
// #[derive(Debug)]
pub struct Tensor {
    // #[pyo3(get)]
    pub input: PyObject,
}

#[pymethods]
impl Tensor {
    #[new]
    fn new(input: &PyBound<PyAny>, py: Python) -> PyResult<Self> {
        let input = input.as_gil_ref();
        let binding = input.get_type().name()?.into_owned();
        let input_type = binding.as_str();
        println!("{}", input_type);
        let output: PyResult<Py<PyAny>> = match input {
            input if binding=="list" => {
                // &PyAny to PyList
                let list = input.downcast::<PyList>()?;
                // &PyList to numpyarry
                // let array: PyBound<'_, PyArray<&PyList, Dim<IxDyn>>>  = PyArray::from_iter_bound(py, list) ;
                // let var =  array? ;// input.downcast()?;
                Ok(list.into_py(py))
            }
            input if binding == "numpy.ndarray" => {
                let list = input.call_method0("tolist")?;
                Ok(list.into_py(py))
            }
            &_ => {

                let err = PyErr::new::<PyTypeError, _>("Unexpecte error accurred");
                Err(err)
            }
        };
        // let out = Box::new(input);
        Ok(Tensor { input: output? })
    }

    // overload `mul`
    // #[]
    fn __mul__(&self, other: PyObject, py: Python) -> PyResult<PyObject> {
        let c = other.bind(py).as_gil_ref();
        let input = self.input.bind(py).as_gil_ref();
        let binding = input.get_type().name()?.into_owned();
        let input_type = binding.as_str();
        match c {
            c if c.is_instance_of::<PyFloat>() && input.is_instance_of::<PyList>() => {
                let var = PyErr::new::<PyTypeError, _>(format!(
                    "for list multiplication, float is not valid, only `int` is valid"
                ));
                return Err(var);
            }
            c if c.is_instance_of::<Tensor>() => {
                let binding: Result<Self, PyErr> = other.extract(py);
                let other: &Py<PyAny> = &binding?.input;
                let result: Py<PyAny> = self
                    .input
                    .call_method1(py, "__mul__", (other,))?
                    .extract(py)
                    .expect("build came with error");
                Ok(Tensor::new(result.bind(py), py).unwrap().into_py(py))
            }
            c if c.is_instance_of::<PyAny>() => {
                let bind = other;
                let result = self
                    .input
                    .call_method1(py, "__mul__", (bind,))
                    .expect("from here");

                Ok(Tensor::new(result.bind(py), py).unwrap().into_py(py))
            }
            _ => {
                let var_name = PyErr::new::<PyTypeError, _>(format!(
                    "TypeError: unsupported operand type for *:'Tensor' and '{}'",
                    type_name_of_val(&other)
                ));
                Err(var_name)
            }
        }
    }

    // String representation
    fn __str__(&self, _py: Python) -> String {
        format!("Tensor({})", self.input)
    }
    fn detach(&self) -> PyObject {
        self.input.clone()
    }
    // Representation for debugging
    fn __repr__(&self) -> String {
        format!("Tensor")
    }
}

// impl clone for `Tensor`
impl Clone for Tensor {
    fn clone(&self) -> Self {
        Tensor {
            input: self.input.clone(),
            // grad: self.grad.clone(),
        }
    }
}

impl TryFrom<PyArrayDyn<npy_float>> for Tensor {
    type Error = ();

    fn try_from(value: PyArrayDyn<npy_float>) -> Result<Self, Self::Error> {
        let py = Python::with_gil(|py| {
            Ok(Tensor {
                input: value.to_object(py),
                // grad: false,
            })
        });
        py
    }
}

impl IntoPyArray for Tensor {
    type Item = npy_float;
    type Dim = IxDyn;
    fn into_pyarray<'py>(self, py: Python<'py>) -> &'py PyArrayDyn<npy_float> {
        let output: &Bound<'py, PyArrayDyn<npy_float>> =
            self.input.downcast_bound(py).expect("Downcast faild : ");
        output.clone().unbind().extract(py).unwrap()
    }
    fn into_pyarray_bound<'py>(
        self,
        py: Python<'py>,
    ) -> Bound<'py, PyArray<Self::Item, Self::Dim>> {
        let output: &Bound<'py, PyArrayDyn<npy_float>> =
            self.input.downcast_bound(py).expect("Downcast faild : ");
        output.clone()
    }
}

// imp `Display` for `Tensor`
impl Display for Tensor {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "({})", self.input)
    }
}
