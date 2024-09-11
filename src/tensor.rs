use std::any::{type_name_of_val, Any};
use std::fmt::{Display, Formatter};

use std::ops::{Add, Mul};

use ndarray::{Dim, IxDynImpl};
use numpy::npyffi::npy_float;
use numpy::{array, dot_bound, npyffi, IntoPyArray, IxDyn, PyArray, PyArrayDyn, ToPyArray};

use pyo3::prelude::*;
use pyo3::Bound as PyBound;
use crate::{ArrayAs, DynDim};

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
use pyo3::types::{PyAny, PyFloat, PyInt};


trait TensorInput<T>{

    fn get_dtype(&self) -> PyResult<T>;
}

impl TensorInput<f32> for PyAny{
    fn get_dtype(&self) -> Result<f32, pyo3::PyErr> {
        self.extract::<f32>()
        
    }
}
impl TensorInput<i32> for PyAny{
    fn get_dtype(&self) -> PyResult<i32> {
        self.extract::<i32>()
        
    }
}


                    

#[pyclass(
    // module = "layer", 
    name = "Tensor", 
    unsendable, 
    subclass, 
    sequence, 
    dict
)]
#[derive(Debug)]
pub struct Tensor {
    #[pyo3(get)]
    input: PyObject,
}

#[pymethods]
impl Tensor {
    #[new]
    fn new(input: &PyBound<PyAny>, py:Python) -> PyResult<Self> {
        let input = input.extract().expect("build came with error");
        Ok(Tensor { input })
    }

    // overload `mul`
    fn __mul__(&self, other: &PyBound<Self>, py:Python) -> PyResult<Self> {
        let other : Py<PyAny> = other.extract().expect("build came with error");
        let result: Py<PyAny> = self.input.call_method1(py, "__mul__", (other,))?.extract(py).expect("build came with error");
        // let res = result.
        Ok(Tensor{ input: result })
        }
    
    // String representation
    fn __str__(&self, py:Python) -> String {
        
        format!("Tensor({})", self.input)
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
        let py = Python::with_gil(|py|
        Ok(Tensor {
            input: value.to_object(py),
            // grad: false,
        }));
        py
    }
}

impl IntoPyArray for Tensor {
    type Item =npy_float;
    type Dim =IxDyn ;
    fn into_pyarray<'py>(self, py: Python<'py>) -> &'py PyArrayDyn<npy_float> {
        let output: &Bound<'py, PyArrayDyn<npy_float>> = self.input.
                                            downcast_bound(py).
                                            expect("Downcast faild : ");
        output.clone().unbind().extract(py).unwrap()
    }
    

    
    fn into_pyarray_bound<'py>(self, py: Python<'py>)
        -> Bound<'py, PyArray<Self::Item, Self::Dim>> {
            let output: &Bound<'py, PyArrayDyn<npy_float>> = self.input.
            downcast_bound(py).
            expect("Downcast faild : ");
            output.clone()
            
    }
}

// impl Into<PyArrayDyn<npy_float>> for Tensor {
//     fn into(self, ) -> PyArrayDyn<npy_float> {
//         let intoo =  
//         self.value.to_owned();
//         let to = Python::with_gil(|py|{
//              intoo.into_pyarray_bound(py).unbind().extract(py).unwrap()

//         });
//         to
//     }
//     // type Into = Bound<Tensor>;
    
// }

// impl From<PyArrayDyn<npy_float>> for Tensor{
//     fn from(value: PyArrayDyn<npy_float>) -> Tensor{
//         let tensor = Python::with_gil(|py|{
//             Tensor::__new__(
//                 &value.into_pyarray_bound(py), Some(false)
//             )

//         } );tensor
//         }
// }





// imp `Display` for `Tensor`
impl Display for Tensor {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result{
        write!(f, "({})", self.input)
        }
}

// imple clone for `Tensor`
// impl Clone for Tensor {
//     fn clone(&self) -> Self {
//         let val = self.input.clone();
//         // let grad = self.grad.clone();
//         Tensor { input: val }
//         }
        
//     fn clone_from(&mut self, source: &Self) {
//         *self = source.clone()
//             }
// }
impl std::fmt::Display for dyn TensorInput<i32> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self)
        // todo!()
    }
}

// impl Into<ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>> for Tensor {
//     fn into(self) -> ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>> {
//         // Access the Python GIL
//         Python::with_gil(|py| {
//             // Extract the PyArray as an ndarray ArrayBase
//             let array: &PyArray<f32, Dim<IxDynImpl>> = self.value.extract(py).unwrap() ;
//             array.to_owned_array()
//         })
//     }
    
// }


// impl Into<ArrayAs<npy_float, OneDim>> for Tensor {
//     fn into(self, ) -> ArrayAs<npy_float, OneDim> {
//         let intoo =  
//         self.value.to_owned();
//         intoo
//         }
//     // type Into = Bound<Tensor>;
    
// }

// impl From<ArrayAs<npy_float, OneDim>> for Tensor{
//     fn from(value: ArrayAs<npy_float, OneDim>) -> Tensor{
//         let tensor = Python::with_gil(|py|{
//             Tensor::__new__(
//                 &value.into_pyarray_bound(py), Some(false)
//             )

//         } );tensor
//         }
// }
// impl `mul` for `Tensodr`
// impl Mul for Tensor {
//     type Output = Tensor;
//     fn mul(self, rhs: Self) -> Self::Output {
//         // Access the Python GIL
//         Python::with_gil(|py| {
//              * rhs.input
//         })
//     }
// }

           
//     });
//     todo!()}
// }
// // impl Add for Tensor {
// //     type Output = Tensor;
// //     fn add(self, rhs: Self) -> Self::Output {
// //         Python::with_gil(|_py|{
// //             self.add(rhs)
// //             })
// //         }
// // }