use crate::{layer::Layers, random_array};
use ndarray::{ArrayBase, ArrayD, Dim, IxDyn, IxDynImpl, OwnedRepr};
use numpy::{IntoPyArray, PyArrayDyn};
use pyo3::{
    prelude::*,
    types::{IntoPyDict, PyDict, PyTuple},
};

pub type Ndarray = ArrayBase<OwnedRepr<f64>, Dim<IxDynImpl>>;
pub type NDArray2 = ArrayBase<OwnedRepr<Vec<Vec<f64>>>, Dim<[usize; 2]>>;
pub type Object = Py<PyArrayDyn<f64>>;

#[derive(FromPyObject)]
#[pyclass(
    module = "nn",
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
    pub weights: PyObject,
    pub bias: PyObject,
    pub is_bias: bool,
    shape: (u16, u16),
}

#[pymethods]
impl Linear {
    #[new]
    #[pyo3(signature = (in_features , out_features, is_bias = true , *args , **kwargs  ))]
    pub fn __new__<'py>(
        py: Python,
        in_features: u16,
        out_features: u16,
        is_bias: Option<bool>,
        args: &Bound<'_, PyAny>,
        kwargs: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<(Self, Layers)> {
        // let super_ = slf.as_super(); // Get &PyRef<BaseClass>
        let is_bias = match is_bias {
            Some(is_bias) => is_bias,
            None => false,
        };

        let random_weight: Ndarray = random_array(in_features.into(), out_features.into());
        let random_bias: Ndarray = if is_bias {
            random_array(1, out_features.into())
        } else {
            ArrayD::zeros(IxDyn(&[1, out_features.into()]))
        };

        let result = (
            Self {
                weights: random_weight.into_pyarray_bound(py).to_owned().into(),
                bias: random_bias.into_pyarray_bound(py).to_owned().into(),
                is_bias: is_bias,
                shape: (in_features, out_features),
            },
            Layers,
        );
        Ok(result)
    }
    // Python::with_gil(|py| {
    //     let random_weight: Ndarray = random_array(in_features.into(), out_features.into());
    //     let random_bias: Ndarray = if is_bias {
    //         random_array(1, out_features.into())
    //     } else {
    //         ArrayD::zeros(IxDyn(&[1, out_features.into()]))
    //     };
    //     Ok(Self {
    //         weights: random_weight.into_pyarray_bound(py).to_owned().into(),
    //         bias: random_bias.into_pyarray_bound(py).to_owned().into(),
    //         is_bias,
    //         shape: (in_features, out_features),
    //     })
    // })

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

    fn __call__(&self, py: Python<'_>, value: &Bound<PyAny>) -> PyResult<PyObject> {
        // Convert value to PyArrayDyn
        let value: &PyArrayDyn<f64> = value.extract()?;

        // Get NumPy module
        let np = PyModule::import_bound(py, "numpy")?;
        // Transpose value
        // let transpose = np.getattr("transpose")?.call1((self.weights.clone(),))?;
        // Perform dot product: self.weights \dot value
        let weight = self.weights.bind(py);
        let bias = self.bias.bind(py);

        let dot_result = np
            .getattr("dot")
            .expect("Invalid import dot")
            .call1((value, weight))
            .expect("dot prod error");

        // Add bias: (self.weights \dot value) + self.bias
        let result = np
            .getattr("add")
            .expect("error import add")
            .call1((dot_result, bias))
            .expect("summation error");

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
        Ok(format!(
            "{}(weights : {} , bias : {})",
            class_name,
            slf.borrow().weights,
            slf.borrow().bias
        ))
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
                ("weighs", slf.borrow().weights.clone()),
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
