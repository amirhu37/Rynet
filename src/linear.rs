
use crate::{add_class, random_bias, random_weight, tensor::Tensor, zero_bias};

use pyo3::{prelude::*, types::PyDict, Bound as PyBound
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
#[derive(FromPyObject)]
// #[derive(Debug, Clone)]
#[pyclass(
    module = "layer",
    name = "Linear",
    // extends= Layers,
    unsendable,
    subclass,
    sequence,
    dict,
    // get_all,
    // set_all
)]

pub struct Linear {
    /// The weights of the linear layer.
    // #[pyo3(set)]
    // #[get]
    pub weight: Tensor,
    /// The bias of the linear layer.
    // #[pyo3(set)]
    // #[pyo3(get)]
    pub bias: Tensor,
    /// Indicates whether the layer uses a bias term.
    // #[pyo3(get)]
    pub is_bias: bool,
    /// Indicates whether the layer is trainable.
    // #[pyo3(get)]
    pub trainable: bool,
    /// The shape of the linear layer as a tuple (in_features, out_features).
    // #[pyo3(get)]
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
    #[pyo3(
        signature = (in_features , out_features, is_bias = true , trainable = true,  *args , **kwargs  ),
        text_signature = "(in_features : int , out_features :int , is_bias: bool = true , trainable: bool = true,  *args , **kwargs  )"

)]
    #[allow(unused_variables)]
    pub fn __new__<'py>(
        py: Python,
        in_features: usize,
        out_features: usize,
        is_bias: Option<bool>,
        trainable: Option<bool>,
        args: &Bound<'_, PyAny>,
        kwargs: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let is_bias = match is_bias {
            Some(is_bias) => is_bias,
            None => false,
        };

        let random_weight =
            random_weight(in_features.into(), out_features.into()).unwrap();
        let random_bias = if is_bias {
            let r_bias = random_bias(out_features.into()).unwrap();
            r_bias
        } else {
            let zero_bias : Tensor =  zero_bias(out_features.into()).unwrap();
            zero_bias
        };

        let result = 
            Self {
                weight: random_weight.into(),
                bias: random_bias.into(),
                is_bias: is_bias,
                trainable: trainable.unwrap_or(true),
                shape: (in_features, out_features),
            };
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

    fn __call__(slf: &Bound<Self>, py: Python<'_>, value: &Bound<Tensor>) -> PyResult<Tensor> {
        //cast `value` in to ndarray
        // let value: &PyArray1<npy_float> = value.extract()?;
        // let value: ArrayAs<npy_float, OneDim> = value.to_owned_array();

        // let weight: &PyArray2<npy_float> = slf.borrow().weight.extract(py)?;
        // let weight: ArrayAs<npy_float, TwoDim> = weight.to_owned_array().t().to_owned();
        let value:Tensor = value.clone().unbind().extract(py).unwrap() ;
        // .extract(py).unwrap() ;
        let doted = value.dot(&slf.borrow().weight, py) ;
        
        // let result: Bound<PyArrayDyn<npy_float>> = dot_bound(
        //     &value.into_pyarray_bound(py),
        //     &weight.into_pyarray_bound(py),
        // )
        // .unwrap();

        let result = doted.add(&slf.borrow().bias, py);
        // println!("{}", result);
        Ok(result)
        // todo!()
    }

    #[getter]
    pub fn weight(slf: &Bound<Self>) -> PyResult<Tensor>{
        Ok(slf.borrow().weight.clone())
    }
    #[getter]
    pub fn bias(slf: &Bound<Self>) -> Tensor{
        slf.borrow().bias.clone()
    }
    fn __str__(slf: &Bound<Self>) -> String {
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

    fn __repr__(slf: &Bound<Self>) -> PyResult<String> {
        // This is the equivalent of `self.__class__.__name__` in Python.
        let class_name = slf.get_type().qualname()?;
        Ok(format!("{}", class_name))
    }

}





#[pymodule]
#[pyo3(name = "linear")]
pub fn linear_module(_py: Python, m: &PyBound<PyModule>) -> PyResult<()> {
    add_class!(m, Linear);
    // m.add_function(wrap_pyfunction!(function_in_module1, m)?)?;
    Ok(())
}