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
use neuaral::Neuaral;
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
pub fn nnmodule(_py: Python, m: &PyBound<PyModule>) -> PyResult<()>{
    add_class!(m, Linear, Neuaral);

    Ok(())
}

#[pymodule]
#[pyo3(name = "loss")]
pub fn lossmodule(_py: Python, m: &Bound<PyModule>) -> PyResult<()>{
    add_class!(m, MSELoss);

    Ok(())
}
#[pymodule]
#[pyo3(name = "layer")]
pub fn layermodule(_py: Python, m: &Bound<PyModule>) -> PyResult<()>{
    add_class!(m, Layer);

    Ok(())
}


#[pymodule]
#[pyo3(name = "rnet")]
pub fn rnet(py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    let module1 = PyModule::new_bound(py, "nnmodule")?;
    let module2 = PyModule::new_bound(py, "layermodule")?;
    let module3 = PyModule::new_bound(py, "lossmodule")?;
   
    m.add_submodule(&module1)?;
    m.add_submodule(&module2)?;
    m.add_submodule(&module3)?;


    add_class!(m, Tensor);
    // add functions
    // add_function!(m, softmax, sigmoid, tanh, relu);

    Ok(())
}
