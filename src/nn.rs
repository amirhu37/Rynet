use crate::{add_class, random_bias, random_weight, ArrayAs, Ndarray, OneDim, TwoDim};
use ndarray::{Array1, ArrayBase, Dim, OwnedRepr};
use pyo3::{types::PyTuple,Bound as PyBound};

use numpy::{dot_bound, npyffi::npy_float, IntoPyArray, PyArray1, PyArray2, PyArrayDyn};
use pyo3::{
    prelude::*,
    types::PyDict,
};
/// Type alias for a 1-dimensional ndarray with owned data and dynamic dimensions.

///# A Python class representing a linear layer in a neural network.
///
///## Attributes:
///*     module (str): The name of the module, which is "layer".
///*     name (str): The name of the class, which is "Linear".
///*     unsendable (bool): Indicates that the class is unsendable.
///*     extends (Layers): Indicates that the class extends the Layers class.
///*     subclass (bool): Indicates that the class can be subclassed.
///*     sequence (bool): Indicates that the class behaves like a sequence.
///*     dict (bool): Indicates that the class has a dictionary attribute.
///*     get_all (bool): Indicates that all attributes are gettable.
///*     set_all (bool): Indicates that all attributes are settable.
// #[derive(FromPyObject)]
#[pyclass(
    module = "nn",
    name = "Linear",
    unsendable,
    // extends= Layers,
    subclass,
    sequence,
    dict,
)]
pub struct Linear {
    /// The weights of the linear layer.
    #[pyo3(get,set, name="weight")]
    pub weight: PyObject,
    /// The bias of the linear layer.
    #[pyo3(get,set, name="bias")]
    pub bias: PyObject,
    /// Indicates whether the layer uses a bias term.
    #[pyo3(get)]
    pub is_bias: bool,
    /// Indicates whether the layer is trainable.
    #[pyo3(get)]
    pub trainable: bool,
    /// The shape of the linear layer as a tuple (in_features, out_features).
    #[pyo3(get)]
    shape: (usize, usize),
}

#[pymethods]
impl Linear {
    #[new]
    #[pyo3( signature = (in_features , out_features, is_bias = true , trainable = true,  *args , **kwargs  ),
    text_signature = "(in_features , out_features, is_bias = true , trainable = true,  *args , **kwargs  )"

            )]
    // #[allow(unused_variables)]
    pub fn __new__<'py>(
        py: Python,
        in_features: usize,
        out_features: usize,
        is_bias: Option<bool>,
        trainable: Option<bool>,
        args: &PyBound<'_, PyAny>,
        kwargs: Option<&PyBound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let is_bias = match is_bias {
            Some(is_bias) => is_bias,
            None => false,
        };

        let random_weight: ArrayAs<f32, TwoDim> =
            random_weight(in_features.into(), out_features.into()).unwrap();
        let random_bias: Ndarray<[usize; 1]> = if is_bias {
            let r_bias: ArrayAs<f32, OneDim> = random_bias(out_features.into()).unwrap();
            r_bias
        } else {
            let zero_bias: ArrayBase<OwnedRepr<f32>, Dim<[usize; 1]>> = Array1::zeros(out_features);
            zero_bias
        };

        let result = 
            Self {
                weight: random_weight.into_pyarray_bound(py).to_owned().into(),
                bias: random_bias.into_pyarray_bound(py).to_owned().into(),
                is_bias,
                trainable: trainable.unwrap_or(true),
                shape: (in_features, out_features),
            };
        Ok(result)
    }

    #[pyo3(text_signature = "($cls )")]
    fn parameters<'py>(slf: &PyBound<Self>, _py: Python<'py>) -> PyResult<Py<PyDict>> {
        // acces dict of the class
        let dict = slf
            .getattr("__dict__")
            .unwrap()
            .downcast::<PyDict>()
            .unwrap()
            .clone();
        let _binding = dict.as_gil_ref().downcast::<PyDict>().unwrap();

        return Ok(dict.unbind());
    }

    fn __call__(slf: &PyBound<Self>, py: Python<'_>, value: &PyBound<PyAny>) -> PyResult<PyObject> {
        //cast `value` in to ndarray
        let value: &PyArray1<npy_float> = value.extract()?;
        let value: ArrayAs<npy_float, OneDim> = value.to_owned_array();
        let weight: &PyArray2<npy_float> = slf.borrow().weight.extract(py)?;
        let weight: ArrayAs<npy_float, TwoDim> = weight.to_owned_array().t().to_owned();

        let result: PyBound<PyArrayDyn<npy_float>> = dot_bound(
            &value.into_pyarray_bound(py),
            &weight.into_pyarray_bound(py),
        )
        .unwrap();

        let result = result.add(slf.borrow().bias.to_owned())?;
        Ok(result.to_object(py))
    }

    fn __str__(slf: &PyBound<Self>) -> String {
        let bias_shape = if !slf.borrow().is_bias {
            0
        } else {
            slf.borrow().shape.1
        };
        let class_name: String = slf.get_type().qualname().unwrap();
        format!(
            "{}(in_features ={}, out_features ={}, is_bias={}, params={}) ",
            class_name,
            slf.borrow().shape.0,
            slf.borrow().shape.1,
            slf.borrow().is_bias,
            slf.borrow().shape.0 * slf.borrow().shape.1 + bias_shape
        )
    }

    fn __repr__(slf: &PyBound<Self>) -> PyResult<String> {
        // This is the equivalent of `self.__class__.__name__` in Python.
        let class_name = slf.get_type().qualname()?;
        Ok(format!("{}", class_name))
    }

}




// #[allow(unconditional_recursion)]
#[derive(
    Debug,
    //  Display,
    Clone,
)]
#[pyclass(module = "nn", unsendable, get_all, set_all, subclass, sequence, dict)]
// #[pyo3(text_signature = "$cls(*args , **kwargs)" )]
// #[display(fmt = "")]
pub struct Neuaral {}
#[pymethods]
impl Neuaral {
    #[new]
    #[pyo3(signature = (*args , **kwargs ) ,)]
    #[allow(unused_variables)]
    pub fn __new__(py: Python, args: &PyBound<PyTuple>, kwargs: Option<&PyBound<PyDict>>) -> Self {
        Neuaral {  } }

    fn parameters<'py>(slf: &PyBound<Self>, _py: Python<'py>) -> PyResult<Py<PyDict>> {
        let dict = slf
            .getattr("__dict__")
            .unwrap()
            .downcast::<PyDict>()
            .unwrap()
            .clone() ;
        let _binding = dict.as_gil_ref().downcast::<PyDict>().unwrap();
        return Ok(dict.unbind());
    }

    pub fn forward(&self, x: PyObject) -> PyResult<PyObject> {
        // TODO
        return Ok(x);
    }
    pub fn __call__(&self, x: PyObject) -> PyResult<PyObject> {
        self.forward(x)
    }
    fn __str__(slf: &PyBound<Self>) -> PyResult<String> {
        let class_name: String = slf.get_type().qualname()?;

        Ok(format!("{}", class_name))
    }

    fn __repr__(slf: &PyBound<Self>) -> PyResult<String> {
        // This is the equivalent of `self.__class__.__name__` in Python.
        let class_name = slf.get_type().qualname()?;
        Ok(format!("{}", class_name))
    }
}
//////////////////////////////////////////////////////////////////////////////////////////////////4
////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////// 
#[pymodule]
#[pyo3(name = "nn")]
pub fn nnmodule(_py: Python, m: &PyBound<PyModule>) -> PyResult<()>{
    add_class!(m, Linear, Neuaral);

    Ok(())
}
