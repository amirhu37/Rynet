use numpy::PyArrayDyn;
use pyo3::prelude::*;

/// A Python class representing a neural network layer.
///
/// Attributes:
///     module (str): The name of the module, which is "rnet".
///     name (str): The name of the class, which is "Layer".
///     unsendable (bool): Indicates that the class is unsendable.
///     subclass (bool): Indicates that the class can be subclassed.
///     sequence (bool): Indicates that the class behaves like a sequence.
///     dict (bool): Indicates that the class has a dictionary attribute.
#[pyclass(module = "layer", name = "Layer", unsendable, subclass, sequence, dict)]
pub struct Layer;

#[pymethods]
impl Layer {
    /// Creates a new instance of the Layers class.
    ///
    /// Args:
    ///     _py (Python): The Python GIL token.
    ///     args (Bound<'_, PyAny>): Positional arguments.
    ///     kwargs (Option<Bound<'_, PyAny>>): Keyword arguments.
    ///
    /// Returns:
    ///     Layers: A new instance of the Layers class.
    #[new]
    #[pyo3(signature = (*args, **kwargs), text_signature = "(*args=None, **kwargs=None)")]
    #[allow(unused_variables)]
    pub fn __new__(
        _py: Python,
        args: &Bound<'_, PyAny>,
        kwargs: Option<&Bound<'_, PyAny>>,
    ) -> Self {
        Layer
    }

    /// Calls the layer with a given value.
    ///
    /// Args:
    ///     _py (Python): The Python GIL token.
    ///     value (Bound<PyAny>): The value to be passed to the layer.
    ///
    /// Returns:
    ///     PyResult<PyObject>: The resulting Python object.
    pub fn __call__(&self, _py: Python, value: &Bound<PyAny>) -> PyResult<PyObject> {
        // Convert value to PyArrayDyn
        let value: &PyArrayDyn<f64> = value.extract()?;
        Ok(value.into())
    }

    /// Returns a string representation of the Layers instance.
    ///
    /// Returns:
    ///     String: A string representation of the Layers instance.
    fn __str__(&self) -> String {
        "Layers instance".to_string()
    }

    /// Returns a detailed string representation of the Layers instance.
    ///
    /// Returns:
    ///     PyResult<String>: A detailed string representation of the Layers instance.
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("Layers()"))
    }

    // #[getter]
    // fn __doc__(&self) -> String {
    //     "
    //     linear Layer. linear Layer. linear Layer.
    //     "
    //     .to_string()
    // }
}
