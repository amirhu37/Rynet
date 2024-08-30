/// interduce file and modules
pub mod functions;
pub mod layer;
pub mod linear;
pub mod loss;
pub mod neuaral;
pub mod optimizers;
pub mod tools;
pub mod tensor;

// use functions::ActiovationFunction;
use layer::Layer;
use linear::Linear;
use loss::MSELoss;
use neuaral::Neural;
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
// pub type d1array = ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>;

/// Type alias for a 2-dimensional ndarray with owned data, where each element is a vector of vectors of f32.
pub type BoundedArrayAs<'py, T> = &'py PyBound<'py, ArrayAs<T, DynDim>>;
/// Type alias for a 2-dimensional ndarray with owned data, where each element is a vector of vectors of f32.
pub type NDArray2 = ArrayBase<OwnedRepr<Vec<Vec<f32>>>, Dim<[usize; 2]>>;
/// Type alias for a Python object that wraps a dynamically-sized ndarray of f32.
pub type Object = Py<PyArrayDyn<f32>>;
pub type BoundedArray<'py> = PyBound<'py, PyArray<f32, IxDyn>>;
pub type PyNdArray<'py, Type, Dimension> = PyBound<'py, PyArray<Type, Dimension>>;
pub type MultiDim = IxDyn;



#[pymodule]
#[pyo3(name = "nn")]
pub fn nn(_py: Python, m: &PyBound<PyModule>) -> PyResult<()>{
    add_class!(m, Linear, Neural);
    println!("mm");
    Ok(())
}

#[pymodule]
#[pyo3(name = "loss")]
pub fn loss_fn(_py: Python, m: &Bound<PyModule>) -> PyResult<()>{
    add_class!(m, MSELoss);

    Ok(())
}
#[pymodule]
#[pyo3(name = "layer")]
pub fn layers(_py: Python, m: &Bound<PyModule>) -> PyResult<()>{
    add_class!(m, Layer);

    Ok(())
}


#[pymodule]
#[pyo3(name = "rnet")]
pub fn rnet(py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    let nn: PyBound<'_, PyModule> = PyModule::new_bound(py, "nn")?;
    let layers: PyBound<'_, PyModule> = PyModule::new_bound(py, "layers")?;
    let loss_fn: PyBound<'_, PyModule> = PyModule::new_bound(py, "loss_fn")?;
   
    m.add_submodule(&nn)?;
    m.add_submodule(&layers)?;
    m.add_submodule(&loss_fn)?;


    // add_class!(m,Linear, Neural, Layer, MSELoss ,Tensor);
    // add functions
    // add_function!(m, softmax, sigmoid, tanh, relu);

    Ok(())
}
