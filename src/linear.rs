use crate::{layer::Layers, random_bias, random_weight, DnArrayAs};
use ndarray::{Array1, ArrayBase, ArrayD, Dim, IxDyn, IxDynImpl, OwnedRepr};

use numpy::{dot_bound, npyffi::npy_float, IntoPyArray, PyArray, PyArrayDyn};
use pyo3::{
    prelude::*,
    types::{IntoPyDict, PyDict, PyTuple},
};
/// Type alias for a 1-dimensional ndarray with owned data and dynamic dimensions.
pub type Ndarray<Dimen> = ArrayBase<OwnedRepr<f32>, Dim<Dimen>>;
// pub type d1array = ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>;

/// Type alias for a 2-dimensional ndarray with owned data, where each element is a vector of vectors of f32.
pub type NDArray2 = ArrayBase<OwnedRepr<Vec<Vec<f32>>>, Dim<[usize; 2]>>;
/// Type alias for a Python object that wraps a dynamically-sized ndarray of f32.
pub type Object = Py<PyArrayDyn<f32>>;
pub type BoundedArray<'py> = Bound<'py, PyArray<f32, IxDyn>>;
pub type PyNdArray<'py, Type, Dimension> = Bound<'py, PyArray<Type, Dimension>>;
pub type MultiDim = IxDyn;
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
    module = "layer",
    name = "Linear",
    unsendable,
    extends= Layers,
    subclass,
    sequence,
    dict,
    get_all,
    set_all
)]

pub struct Linear {
    /// The weights of the linear layer.
    pub weight: PyObject,
    /// The bias of the linear layer.
    pub bias: PyObject,
    /// Indicates whether the layer uses a bias term.
    pub is_bias: bool,
    /// Indicates whether the layer is trainable.
    pub trainable: bool,
    /// The shape of the linear layer as a tuple (in_features, out_features).
    shape: (usize, usize),
}

#[pymethods]
impl Linear {
    ///## Creates a new instance of the Linear class.
    ///
    ///### Args:
    ///*     py (Python): The Python GIL token.
    ///*     in_features (u16): The number of input features.
    ///*     out_features (u16): The number of output features.
    ///*     is_bias (Option<bool>): Whether the layer uses a bias term.
    ///*     trainable (Option<bool>): Whether the layer is trainable.
    ///*     args (Bound<'_, PyAny>): Positional arguments.
    ///*     kwargs (Option<Bound<'_, PyAny>>): Keyword arguments.
    ///
    /// Returns:
    ///     PyResult<(Self, Layers)>: A new instance of the Linear class and its base Layers class.
    #[new]
    #[pyo3(signature = (in_features , out_features, is_bias = true , trainable = true,  *args , **kwargs  ))]
    #[allow(unused_variables)]
    pub fn __new__<'py>(
        py: Python,
        in_features: usize,
        out_features: usize,
        is_bias: Option<bool>,
        trainable: Option<bool>,
        args: &Bound<'_, PyAny>,
        kwargs: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<(Self, Layers)> {
        let is_bias = match is_bias {
            Some(is_bias) => is_bias,
            None => false,
        };

        let random_weight: Ndarray<IxDynImpl> =
            random_weight(in_features.into(), out_features.into());
        let random_bias: Ndarray<[usize; 1]> = if is_bias {
            let r_bias: ArrayBase<OwnedRepr<f32>, Dim<[usize; 1]>> =
                random_bias(out_features.into());
            r_bias
        } else {
            let zero_bias: ArrayBase<OwnedRepr<f32>, Dim<[usize; 1]>> = Array1::zeros(out_features);
            zero_bias
        };

        let result = (
            Self {
                weight: random_weight.into_pyarray_bound(py).to_owned().into(),
                bias: random_bias.into_pyarray_bound(py).to_owned().into(),
                is_bias: is_bias,
                trainable: trainable.unwrap_or(true),
                shape: (in_features, out_features),
            },
            Layers,
        );
        Ok(result)
    }

    #[pyo3(text_signature = "($cls )")]
    fn parameters<'py>(slf: &Bound<Self>, _py: Python<'py>) -> Py<PyDict> {
        // acces dict of the class
        let dict = slf
            .getattr("__dict__")
            .unwrap()
            .downcast::<PyDict>()
            .unwrap()
            .clone();
        let _binding = dict.as_gil_ref().downcast::<PyDict>().unwrap();

        return dict.unbind();
    }

    fn __call__(slf: &Bound<Self>, py: Python<'_>, value: &Bound<PyAny>) -> PyResult<PyObject> {
        //cast `value` in to ndarray
        let value: &PyArrayDyn<npy_float> = value.extract()?;
        let value: DnArrayAs<npy_float> = value.to_owned_array();
        let weight: &PyArrayDyn<npy_float> = slf.borrow().weight.extract(py)?;
        let weight: DnArrayAs<npy_float> = weight.to_owned_array().t().to_owned();

        let result: Bound<PyArrayDyn<npy_float>> = dot_bound(
            &value.into_pyarray_bound(py),
            &weight.into_pyarray_bound(py),
        )
        .unwrap();

        let result = result.add(slf.borrow().bias.to_owned())?;
        Ok(result.to_object(py))
    }

    fn __str__(slf: &Bound<Self>) -> String {
        let bias_shape = if !slf.borrow().is_bias {
            0
        } else {
            slf.borrow().shape.1
        };
        let class_name = slf.get_type().qualname().unwrap();
        format!(
            "{}(in = {},out = {}, params={}) ",
            class_name,
            slf.borrow().shape.0,
            slf.borrow().shape.1,
            slf.borrow().shape.0 * slf.borrow().shape.1 + bias_shape
        )
    }

    fn __repr__(slf: &Bound<Self>) -> PyResult<String> {
        // This is the equivalent of `self.__class__.__name__` in Python.
        let class_name = slf.get_type().qualname()?;
        Ok(format!("{}", class_name))
    }

    #[getter]
    fn __doc__(&self) -> String {
        format!(
            "
        linear Layer. linear Layer. linear Layer. 
        "
        )
    }
    fn __iter__(slf: &Bound<Self>) -> PyObject {
        let class_name = slf.get_type().qualname().unwrap();
        Python::with_gil(|py| {
            // let list = PyList::new_bound(py, slf.borrow().weights.clone().to_object(py));
            let locals = [
                ("weighs", slf.borrow().weight.clone()),
                ("self", class_name.to_object(py)),
                ("bias", slf.borrow().bias.clone()),
            ]
            .into_py_dict_bound(py);
            let result = py
                .eval_bound("list(self.__dict__.values())", None, Some(&locals))
                .unwrap()
                .unbind();
            let py_obj: PyObject = result.downcast_bound(py).unwrap().clone().unbind();
            py_obj
        })
    }
}
