/// interduce file and modules
pub mod functions;
pub mod layer;
pub mod linear;
pub mod loss;
pub mod neuaral;
pub mod optimizers;
pub mod tools;

/// import files and modules
use functions::*;
use layer::Layers;
use linear::Linear;
use loss::*;
use neuaral::Neuaral;

use ndarray::{Array1, ArrayBase, ArrayD, Dim, IxDyn, IxDynImpl, OwnedRepr};
use numpy::{ PyArray, PyArrayDyn};
#[allow(unused_imports)]
use pyo3::Bound as PyBound;
use pyo3::*;
use pyo3::{
    pymodule,
    types::{IntoPyDict, PyDict, PyModule},
    Py, PyResult, Python,
};

use rand::Rng;

pub type NpNdarrayAs<T> = Py<PyArray<T, Dim<IxDynImpl>> >;

pub type DnArrayAs<T> = ArrayBase<OwnedRepr<T>, Dim<IxDynImpl>>;
pub type N1ArrayAs<T> = ArrayBase<OwnedRepr<T>, Dim<[usize; 1]>>;
pub type N2ArrayAs<T> = ArrayBase<OwnedRepr<T>, Dim<[usize; 2]>>;
pub type N3ArrayAs<T> = ArrayBase<OwnedRepr<T>, Dim<[usize; 3]>>;

pub type BoundedArrayAs<'py, T> = &'py PyBound<'py, DnArrayAs<T>>;
pub type Ndarray<Dimen> = ArrayBase<OwnedRepr<f32>, Dim<Dimen>>;
/// Type alias for a 2-dimensional ndarray with owned data, where each element is a vector of vectors of f32.
pub type NDArray2 = ArrayBase<OwnedRepr<Vec<Vec<f32>>>, Dim<[usize; 2]>>;
/// Type alias for a Python object that wraps a dynamically-sized ndarray of f32.
pub type Object = Py<PyArrayDyn<f32>>;
pub type BoundedArray<'py> = PyBound<'py, PyArray<f32, IxDyn>>;
pub type PyNdArray<'py, Type, Dimension> = PyBound<'py, PyArray<Type, Dimension>>;
pub type MultiDim = IxDyn;




fn random_weight(n: usize, m: usize) -> PyResult<DnArrayAs<f32>> {
    let mut rng = rand::thread_rng();
    let mut array: DnArrayAs<f32> = ArrayD::zeros(IxDyn(&[n, m]));
    // array.t()
    for i in 0..n {
        for j in 0..m {
            array[[i, j]] = rng.gen::<f32>();
        }
    }
    Ok(array)
}
fn random_bias<'py>(n: usize) -> PyResult<N1ArrayAs<f32>> {
    let mut rng = rand::thread_rng();

    let mut array: N1ArrayAs<f32> = Array1::zeros(n);
    for i in 0..n {
        array[i] = rng.gen::<f32>();
    }
    Ok(array)
}

pub fn _py_run(value: &Bound<PyAny>, command: &str) -> PyResult<Py<PyDict>> {
    Python::with_gil(|py| {
        let locals = [("value", value)].into_py_dict_bound(py);
        let result = py.eval_bound(command, None, Some(&locals))?.unbind();
        let py_dict = result.downcast_bound(py).unwrap().clone().unbind();
        Ok(py_dict.into())
    })
}



#[macro_export]
macro_rules! add_class {
    ($module : ident , $($class : ty), +) => {
        $(
            $module.add_class::<$class>()?;
        )+

    };
}
macro_rules! add_function {
    ($module : ident , $($function : ident), +) => {
        $(
           $module.add_wrapped(wrap_pyfunction!($function))?;
        )+
    };
}
#[pymodule]
#[pyo3(name = "rnet")]
pub fn nnet(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    add_class!(m, Linear, Neuaral, Layers, ActiovationFunction, MSELoss);
    // add functions
    add_function!(m, softmax, sigmoid, tanh, relu);
    // add_function!(m, cross_entropy);
    // add_function!(m, mse);
    Ok(())
}
