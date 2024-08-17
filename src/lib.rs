/// interduce file and modules
pub mod functions;
pub mod layer;
pub mod linear;
pub mod loss;
pub mod neuaral;
pub mod optimizers;
pub mod tools;
pub mod tensor;

/// import files and modules
use functions::*;

use ndarray::{Array1, Array2, ArrayBase, ArrayD, Dim, IxDyn, IxDynImpl, OwnedRepr};
use numpy::{IntoPyArray, Ix2, PyArray, PyArrayDyn};
#[allow(unused_imports)]
use pyo3::Bound as PyBound;
use pyo3::*;
use pyo3::{ 
    pymodule,
    types::{IntoPyDict, PyDict, PyModule},
    Py, PyResult, Python,
};

use rand::Rng;
use tensor::Tensor;

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

fn random_weight(n: usize, m: usize) -> PyResult<Tensor > {
    let mut rng = rand::thread_rng();
    let mut array: ArrayAs<f32, DynDim> = ArrayD::zeros( IxDyn(&[n,m]) );
    // array.t()
    for i in 0..n {
        for j in 0..m {
            array[[i, j]] = rng.gen::<f32>();
        }
    }
    let array = Python::with_gil(|py|{
        let array = Tensor::__new__(&array.into_pyarray_bound(py), Some(false));
        array
    });

    Ok(array)
}


fn random_bias<'py>(n: usize) -> PyResult<Tensor > {
    let mut rng = rand::thread_rng();

    let mut array: ArrayAs<f32, DynDim> = ArrayD::zeros(IxDyn(&[n,0]));
    for i in 0..n {
        array[i] = rng.gen::<f32>();
    }
    
    let array = Python::with_gil(|py|{
        let array = Tensor::__new__(&array.into_pyarray_bound(py), Some(false));
        array
    });

    Ok(array)
}

fn zero_bias<'py>(n: usize) -> PyResult<Tensor > {
    let mut array: ArrayAs<f32, DynDim> = ArrayD::zeros(IxDyn(&[n,0]));    
    let y = Python::with_gil(|py|{
        let array = Tensor::__new__(&array.into_pyarray_bound(py), Some(false));
        array
    });

    Ok(y)
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

#[macro_export]
macro_rules! add_sub_module {
    ($module : ident , $($class : ty), +) => {
        $(
            $module.add_sub_module::<$class>()?;
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
pub fn nnet(py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    let linearmodule = PyModule::new_bound(py, "linear_module")?;
    let layermodule = PyModule::new_bound(py, "layer_module")?;
    let lossmodule = PyModule::new_bound(py, "loss_module")?;
    let neuaralmodule = PyModule::new_bound(py, "neural_module")?;
    
    // let activationmodule = PyModule::new_bound(py, "activation_module")?;
    // let optimizemodule = PyModule::new_bound(py, "optimizer_module")?;

    let _ = m.add_submodule(&layermodule);
    let _ = m.add_submodule(&linearmodule);
    let _ = m.add_submodule(&lossmodule);
    let _ = m.add_submodule(&neuaralmodule);
    add_class!(m, Tensor);
    add_function!(m, softmax, sigmoid, tanh, relu);

    Ok(())
}
