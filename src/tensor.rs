use std::any::{type_name_of_val, Any};
use std::fmt::{Display, Formatter};

use std::ops::{Add, Mul};

use ndarray::{Dim, IxDynImpl};
use numpy::{array, dot_bound, IntoPyArray, PyArray, ToPyArray};
use numpy::{npyffi::npy_float, PyArrayDyn};
use pyo3::prelude::*;
// use pyo3::Bound as PyBound;
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
use pyo3::types::{PyAny, PyInt};



#[derive(Debug, Clone)]
enum TensorInput {
    Int(i64),
    Float(f32),
    Long(f64),
    IntVec(PyObject),
    FloatVec(PyObject),
    LongVec(PyObject)
}
impl TensorInput {
    fn extract_value(&self, py : Python)->PyObject{
        match self {

            TensorInput::Int(x) => x.to_object(py),
            TensorInput::Float(x) => x.to_object(py) ,
            TensorInput::Long(x) => x.to_object(py),
            TensorInput::IntVec(x) => x.to_object(py),
            TensorInput::FloatVec(x) => x.to_object(py),
            TensorInput::LongVec(x) => x.to_object(py),
    }
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

pub struct Tensor {
    // #[pyo3(get)]
    input: TensorInput,
}

#[pymethods]
impl Tensor {
    #[new]
    fn new(input: &PyAny) -> PyResult<Self> {
        if let Ok(val) = input.extract::<i64>() {
            Ok(Tensor {
                input: TensorInput::Int(val),
            })
        } else if let Ok(val) = input.extract::<f32>() {
            Ok(Tensor {
                input: TensorInput::Float(val),
            })
        }else if let Ok(val) = input.extract::<f64>() {
            Ok(Tensor {
                input: TensorInput::Long(val),
            })
        } else if let Ok(val) = input.extract::<PyObject>() {
            Ok(Tensor {
                input: TensorInput::IntVec(val),
            })
        } else if let Ok(val) = input.extract::<PyObject>() {
            Ok(Tensor {
                input: TensorInput::FloatVec(val),
            })
        }else if let Ok(val) = input.extract::<PyObject>() {
            Ok(Tensor {
                input: TensorInput::LongVec(val),
            })
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "Input should be an int, float, or a vector of int/float",
            ))
        }
    }
    // #[get]
    fn get_input(&self, py:Python)->PyResult<PyObject>{
        let input = self.input.clone();
        match input {
            TensorInput::Int(val) => Ok(val.to_object(py)),
            TensorInput::Float(val) => Ok(val.to_object(py)) ,
            TensorInput::Long(val) => Ok(val.to_object(py)),
            TensorInput::IntVec(val) => Ok(val.to_object(py)),
            TensorInput::FloatVec(val) => Ok(val.to_object(py)),
            TensorInput::LongVec(val) => Ok(val.to_object(py)),
        }
    }

    // String representation
    fn __str__(&self) -> String {
        
        format!("Tensor({:?}, {})", self.input, type_name_of_val(&self.input))
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

        write!(f, "({})", self.input)
        }
}

// imple clone for `Tensor`
impl Clone for Tensor {
    fn clone(&self) -> Self {
        let val = self.input.clone();
        // let grad = self.grad.clone();
        Tensor { input: val }
        }
}
impl std::fmt::Display for TensorInput {
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
impl Mul for Tensor {
    type Output = Tensor;
    fn mul(self, rhs: Self) -> Self::Output {
        // Access the Python GIL
        Python::with_gil(|py| {
            // Extract the PyArray as an ndarray ArrayBase or int ot float
            // check input type
            let left = self.input.extract_value(py);
            let right = rhs.input.extract_value(py);
            // check if both are int or float
            if type_name_of_val(&left) == "PyInt"  && type_name_of_val(&right) == "PyInt" {
                let tensor_input = &self.input;
                let extract_value = tensor_input.extract_value(py);
                let left: &Bound<'_, PyInt> = extract_value.downcast_bound(py).unwrap().downcast_exact::<PyInt>().unwrap() ;
                let tensor_input = &rhs.input;
                let extract_value = tensor_input.extract_value(py);
                let right: &Bound<'_, PyInt> = extract_value.downcast_bound(py).unwrap().downcast_exact::<PyInt>().unwrap() ;
                // Perform the multiplication
                let result = left.mul(right) ;
                // Create a new Tensor with the result
                // Tensor::new(TensorInput::Int(result))
                //  left * right
                }else if type_name_of_val(&left) == "PyFloat"  && type_name_of_val(&right) == "PyFloat" {todo!()
                }else{}
        
            // let left: &Bound<'_, PyFloat> = self.input.extract_value(py).down
            // let right: &Bound<'_, PyFloat> = rhs.input.extract_value(py).down


           
    });
    todo!()}
}
// impl Add for Tensor {
//     type Output = Tensor;
//     fn add(self, rhs: Self) -> Self::Output {
//         Python::with_gil(|_py|{
//             self.add(rhs)
//             })
//         }
// }