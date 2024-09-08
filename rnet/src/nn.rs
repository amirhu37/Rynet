use pyo3::prelude::*;
use pyo3::Bound as PyBound;
use crate::add_class;
use crate::linear::Linear;
use crate::model::Model;

#[pymodule]
#[pyo3(name = "nn")]
pub fn register(module: &PyBound<PyModule> ) -> PyResult<()> {

    add_class!(module,Linear, Model);

    Ok(())
}
