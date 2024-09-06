/// interduce file and modules
pub mod functions;
pub mod layer;
pub mod linear;
// mod nn;
pub mod loss_functions;
pub mod model;
pub mod optimizers;
pub mod tools;
pub mod tensor;
pub mod types;

use linear::Linear;
use model::Model;
// use functions::ActiovationFunction;
// use layer::Layer;
// use crate::nn::linear::Linear;
// use crate::nn::model::Model;
// use loss_functions::MSELoss;
// use model::Model;
// use prelude::PyModuleMethods;
use tensor::Tensor;

use ndarray::{ArrayBase, Dim, IxDyn, IxDynImpl, OwnedRepr};
use numpy::{PyArray, PyArrayDyn};
#[allow(unused_imports)]
use pyo3::Bound as PyBound;
use pyo3::*;
use pyo3::{
    pymodule,
    types::PyModule,
    Py, PyResult, Python,
};


// use rand::Rng;
pub type DynDim = Dim<IxDynImpl>;
pub type OneDim = Dim<[usize; 1]>;
pub type TwoDim = Dim<[usize; 2]>;
pub type ThreeDim = Dim<[usize; 3]>;

pub type NpNdarrayAs<T, D> = Py<PyArray<T, D>>;

pub type ArrayAs<T, D> = ArrayBase<OwnedRepr<T>, D>;

pub type Ndarray<Dimen> = ArrayBase<OwnedRepr<f32>, Dim<Dimen>>;
/// Type alias for a 2-dimensional ndarray with owned data, where each element is a vector of vectors of f32.
pub type BoundedArrayAs<'py, T> = &'py PyBound<'py, ArrayAs<T, DynDim>>;
/// Type alias for a 2-dimensional ndarray with owned data, where each element is a vector of vectors of f32.
pub type NDArray2 = ArrayBase<OwnedRepr<Vec<Vec<f32>>>, Dim<[usize; 2]>>;
/// Type alias for a Python object that wraps a dynamically-sized ndarray of f32.
pub type Object = Py<PyArrayDyn<f32>>;
pub type BoundedArray<'py> = PyBound<'py, PyArray<f32, IxDyn>>;
pub type PyNdArray<'py, Type, Dimension> = PyBound<'py, PyArray<Type, Dimension>>;
pub type MultiDim = IxDyn;


// #[pymodule]
// #[pyo3(name = "nn")]
// pub fn nn_(py: Python, module: &PyBound<PyModule> ) -> PyResult<()> {
//     add_class!(module,Linear, Model);
//     Ok(())
// }

#[pymodule]
#[pyo3(name = "rnet")]
pub fn rnet(py: Python, module: &PyBound<PyModule> ) -> PyResult<()> {
    add_class!(module, Tensor);

    let nn: PyBound<'_, PyModule> = PyModule::new_bound(py, "nn")?;
    add_class!(nn,Linear, Model);
    module.add_submodule(&nn)?;

    Ok(())
}



// #[pyo3(name = "rnet")]
// pub fn rnet(py: Python, module: &PyBound<PyModule> ) -> PyResult<()> {
//     // let layers: PyBound<'_, PyModule> = PyModule::new_bound(py, "layers_")?;
//     // let loss_: PyBound<'_, PyModule> = PyModule::new_bound(py, "loss_")?;
   
//     // module.add_submodule(&nn)?;
//     // module.add_submodule(&layers)?;
//     // module.add_submodule(&loss_)?;

//     module. add_class::<Linear>()?;
//     // add_class!(module,Linear, Model, Layer, MSELoss ,Tensor);
//     // add functions
//     // add_function!(m, softmax, sigmoid, tanh, relu);
