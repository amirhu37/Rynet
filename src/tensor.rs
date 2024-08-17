use std::clone;
use std::fmt::{Display, Formatter};
use std::ops::{Add, Deref, Mul};

use ndarray::linalg::Dot;
use ndarray::{Array, ArrayBase, ArrayViewD, Dim, IxDynImpl, OwnedRepr};
use numpy::{dot_bound, IntoPyArray, PyArray, PyArrayMethods, ToPyArray};
use numpy::{npyffi::npy_float, IxDyn, PyArrayDyn};
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use pyo3::Bound as PyBound;
use crate::{ArrayAs, DynDim, OneDim, ThreeDim, TwoDim};

// Define the TensorTrait trait
pub trait TensorTrait {
    // We can define some common behavior here if needed
}

// Implement TensorTrait for ArrayAs<npy_float, OneDim>
impl TensorTrait for ArrayAs<npy_float, OneDim> {}

// Implement TensorTrait for ArrayAs<npy_float, TwoDim>
impl TensorTrait for ArrayAs<npy_float, TwoDim> {}

// Implement TensorTrait for ArrayAs<npy_float, ThreeDim>
impl TensorTrait for ArrayAs<npy_float, ThreeDim> {}

#[pyclass(
    module = "layer", 
    name = "Layer", 
    unsendable, 
    subclass, 
    sequence, 
    dict
)]
// #[derive(Debug, FromPyObject)]
// #[derive(FromPyObject)]
pub struct Tensor {
    #[pyo3(get)]
    value: Py<PyArrayDyn<npy_float>>, 
    #[pyo3(get)]
    grad : bool
}



#[pymethods]
impl Tensor {
    // New method that creates a Tensor instance
    #[new]
    pub fn __new__(
        // py: Python,
        value: &PyBound<PyAny>,
        req_grad: Option<bool>,
    ) -> Self {
        let value: &PyArrayDyn<f32> = value.extract::<&PyArrayDyn<npy_float>>().unwrap();
        // let shape = value.shape()
        let value: Py<PyArrayDyn<f32>> = value.extract().unwrap() ;
        Tensor { 
            value: value, 
            // shape : shape,
            grad: req_grad.unwrap_or(false),
        }
    }

    pub fn dot(&self, other: &Tensor, py: Python) -> Tensor {
        let value : &PyArrayDyn<npy_float> = self.value.extract(py).unwrap();
        let value: ArrayAs<npy_float, DynDim> = value.to_owned_array();
        let other_value : &PyArrayDyn<npy_float> = other.value.extract(py).unwrap();
        let other_value: ArrayAs<npy_float, DynDim> = other_value.to_owned_array();
    
        // Compute the dot product using `ndarray`'s dot method
        let result:Bound<PyArrayDyn<npy_float>> = dot_bound( &value.into_pyarray_bound(py),
         &other_value.into_pyarray_bound(py)).unwrap();
    
        // Convert the result back to a PyArray and wrap it in a Py<PyArray>
        Tensor {
            value: result.unbind(),
            grad: self.grad || other.grad,
        }
    }
    

    // `Add` method
    pub fn add(&self, other: &Tensor, py : Python) -> Tensor {
        let value : &PyArrayDyn<npy_float> = self.value.extract(py).unwrap();
        let value: ArrayAs<npy_float, DynDim> = value.to_owned_array();
        let other_value : &PyArrayDyn<npy_float> = other.value.extract(py).unwrap();
        let other_value: ArrayAs<npy_float, DynDim> = other_value.to_owned_array();
        let result: Py<PyArrayDyn<npy_float>> = value.to_pyarray_bound(py)
                                        .add(&other_value
                                            .to_pyarray_bound(py)
                                        ).unwrap().downcast().unwrap().clone().unbind() ;
        Tensor {
            value: result,
            grad: self.grad || other.grad,
        }
    }

    // Method to transpose the tensor
    pub fn transpose(&self, py : Python) -> Tensor {
        let transposed_data : &PyArrayDyn<npy_float> = self.value.extract(py).unwrap()  ;
        let transposed_data: ArrayAs<npy_float, DynDim> = transposed_data.to_owned_array().reversed_axes();
        Tensor {
            value: transposed_data.to_pyarray_bound(py).unbind(),
            grad: self.grad,
        }
    }

    // String representation
    fn __str__(&self) -> String {
        
        format!("Tensor({}, grad = {})", self.value , self.grad)
    }

    // Representation for debugging
    fn __repr__(&self) -> String {
        format!("Tensor")
    }
}

//impl clone for `Tensor`
// impl Clone for Tensor {
//     fn clone(&self) -> Self {
//         Tensor {
//             value: self.value.clone(),
//             grad: self.grad.clone(),
//             }
//         }
// }

// impl From<PyArrayDyn<npy_float>> for Tensor {
//     fn from(value: PyArrayDyn<npy_float>) -> Tensor {
//         Tensor {
//             value,
//             grad: false,
//         }
//     }
// }

// impl IntoPyArray<'_> for Tensor {
//     fn into_pyarray<'py>(self, py: Python<'py>) -> &'py PyArrayDyn<npy_float> {
//         self.value.into_pyarray(py)
//     }
// }
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
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result
    {

        write!(f, "({}, grad_bound = {})", self.value, self.grad)
        }
}

// imple clone for `Tensor`
impl Clone for Tensor {
    fn clone(&self) -> Self {
        let val = self.value.clone();
        let grad = self.grad.clone();
        Tensor { value: val, grad: grad }
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
impl Mul for Tensor {
    type Output = Tensor;
    fn mul(self, rhs: Self) -> Self::Output {
        Python::with_gil(|py|{
            self.dot(&rhs, py)
        })
    }
}
impl Add for Tensor {
    type Output = Tensor;
    fn add(self, rhs: Self) -> Self::Output {
        Python::with_gil(|py|{
            self.add(rhs)
            })
        }
}