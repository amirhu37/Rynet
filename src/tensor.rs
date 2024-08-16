use std::fmt::{Display, Formatter};

use ndarray::linalg::Dot;
use ndarray::Array;
use numpy::{dot_bound, IntoPyArray, PyArrayMethods, ToPyArray};
use numpy::{npyffi::npy_float, IxDyn, PyArrayDyn};
use pyo3::prelude::*;
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
#[derive(Debug, Clone)]
pub struct Tensor {
    value: ArrayAs<npy_float, IxDyn>,  // Use Box<dyn TensorTrait> for trait objects
    grad : bool
}


#[pymethods]
impl Tensor {
    // New method that creates a Tensor instance
    #[new]
    pub fn __new__(
        // py: Python,
        value:  &PyBound<PyArrayDyn<npy_float>>,
    req_grad : Option<bool> ) -> Self {
        Tensor { 
            value: value.to_owned_array(), 
            grad : req_grad.unwrap_or(false)
         }
    }

        // Method to calculate the dot product of two tensors
    pub fn dot(&self, other: &Tensor, py: Python) -> Tensor {
        let value: ArrayAs<npy_float, DynDim> = self.value.to_owned();
        let other: ArrayAs<npy_float, DynDim> = other.value.to_owned();
        let result: ArrayAs<npy_float, DynDim>  = value * other;
        // let result = dot_bound(&value.into_pyarray_bound(py),
        //                                  &other.into_pyarray_bound(py))
        //                                  .unwrap();
        
        // let owned_array = result.to_owned_array();
        // Tensor::__new__(&result, Some(false))
        result.into()
    }
    // `Add` method
    pub fn add(&self, other: &Tensor, py: Python) -> Tensor {
        let value: ArrayAs<npy_float, DynDim> = self.value.to_owned();
        let other: ArrayAs<npy_float, DynDim> = other.value.to_owned();
        let s = value + other;
        s.into()
    }
    // Method to transpose the tensor
    pub fn transpose(&self, py: Python) -> Tensor {
        let binding = self.value.to_owned();
        let transposed_data = binding.t();
        Tensor::__new__(&transposed_data.to_pyarray_bound(py), Some(false))
        // Tensor::__new__(transposed_data, Some(false))
    }
    fn __str__(&self)->String{
        

        format!("({}, grad_bound = {})", self.value, self.grad)
    }
    fn __repr__(slf: &Bound<Self>)->String{
        format!("Tensor({})", slf.borrow().value)
    }
}

impl Into<ArrayAs<npy_float, DynDim>> for Tensor {
    fn into(self, ) -> ArrayAs<npy_float, DynDim> {
        let intoo =  
        self.value.to_owned();
        intoo
        }
    // type Into = Bound<Tensor>;
    
}

impl From<ArrayAs<npy_float, DynDim>> for Tensor{
    fn from(value: ArrayAs<npy_float, DynDim>) -> Tensor{
        let tensor = Python::with_gil(|py|{
            Tensor::__new__(
                &value.into_pyarray_bound(py), Some(false)
            )

        } );tensor
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
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result
    {

        write!(f, "({}, grad_bound = {})", self.value, self.grad)
        }
}

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