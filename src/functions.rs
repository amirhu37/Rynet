#[allow(unused_variables)]
use numpy::{IntoPyArray, PyArrayDyn};
use pyo3::{Py, PyAny, PyResult, Python};
// use ActiovationFunction::*;
use ndarray::ArrayD;
use pyo3::prelude::*;

use crate::NpNdarrayAs;

#[allow(dead_code)]
// trait ActiovationFunction {
//     fn dfx(y: fn(f64)) -> f64;
//     fn fx(x: fn(f64)) -> f64;
// }

// #[derive(FromPyObject)]
#[pyclass]
pub struct ActiovationFunction {
    fx: fn(Py<PyAny>) -> Py<PyAny>,
    df: fn(Py<PyAny>) -> Py<PyAny>,
}

#[pyfunction]
pub fn sigmoid(x: Bound<PyAny>) -> NpNdarrayAs<f64> {
    let func = |value: f64| 1.0 / (1.0 + (-value).exp());
    let y: NpNdarrayAs<f64> = apply_func(&x, func).unwrap();
    y
}
#[pyfunction]
pub fn tanh(x: Bound<PyAny>) -> NpNdarrayAs<f64> {
    let func = |value: f64| (2.0 / (1.0 + (-value).exp())).tanh();
    let y: NpNdarrayAs<f64> = apply_func(&x, func).unwrap();
    y
}
#[pyfunction]
pub fn relu(x: Bound<PyAny>) -> NpNdarrayAs<f64> {
    let func = |value: f64| if value > 0.0 { value } else { 0.0 };
    let y: NpNdarrayAs<f64> = apply_func(&x, func).unwrap();
    y
}
#[pyfunction]
pub fn softmax(x: Bound<PyAny>) -> NpNdarrayAs<f64> {
    let func = |value: f64| 1.0 / (1.0 + (-value).exp());
    let y: NpNdarrayAs<f64> = apply_func(&x, func).unwrap();
    y
}

#[allow(dead_code)]
pub fn apply_func(input: &Bound<PyAny>, func: impl Fn(f64) -> f64) -> PyResult<NpNdarrayAs<f64>> {
    // Convert the input to a NumPy array
    let input_array: &PyArrayDyn<f64> = input.extract()?;
    let res: NpNdarrayAs<f64> = Python::with_gil(|py| {
        // Convert the NumPy array to an ndarray ArrayD
        let input_array: numpy::PyReadonlyArray<f64, ndarray::Dim<ndarray::IxDynImpl>> =
            input_array.readonly();
        let input_array: ndarray::ArrayBase<
            ndarray::ViewRepr<&f64>,
            ndarray::Dim<ndarray::IxDynImpl>,
        > = input_array.as_array();

        // Apply the function to each element of the array
        let result: ArrayD<f64> = input_array.mapv(func);
        let bound = result.into_pyarray_bound(py);
        let c = bound.to_owned();
        c.unbind()
    });
    // Convert the result back to a NumPy array and return
    Ok(res)
}
#[allow(dead_code)]
fn funcs() {
    #[allow(unused_variables)]
    let f_sigmoid = |x: f64| 1.0 / (1.0 + (-x).exp());
    // derivative of sigmoid
    #[allow(unused_variables)]
    let df_dsigmoid = |y: f64| y * (1.0 - y);

    // sfotmax and its derivative
    // let f_softmax = |x : f64| { x.exp() / (x.exp().sum() )};
    #[allow(unused_variables)]
    let df_dsoftmax = |y: f64| y * (1.0 - y);
}

use numpy::{IxDyn, PyArrayDyn};
use pyo3::{ffi::PyObject, pyclass, pymethods, types::PyModule, PyAny, Python};
use ActiovationFunction::*;

use ndarray::{ArrayBase, ArrayD, Dim, IxDynImpl, OwnedRepr};

use crate::darrayf64;
// #[pyclass]
struct Sigmoid{
    function: fn(f64),
    derivation: fn(f64),
}

// #[pymethods]
impl Sigmoid {
    // #[new]
    // #[staticmethod]
    fn __new__() -> Self {
        let f = |x : f64| { 1.0 / (1.0 + (-x).exp())};
        let df = |x: f64| { f(x) * (1.0 - f(x))};
todo!()
}

}

// static CODE = r"#

// #";

fn sigmoid(x :PyAny){
    Python::with_gil(|py| {
        let value: &PyArrayDyn<f64> = x.extract().unwrap();

        const CODE: &str = r"#from numpy import shape
        function = lambda x : shape(x)";
        let module = PyModule::from_code(py, CODE, "", "").unwrap();
        let fun = module.getattr("function").unwrap();
        let args = ("hello",);
        let result = fun.call1(args).unwrap();
        // let s = (1,2);
        // let shape = result.extract::<&(u16, u16)>().unwrap();
        // assert_eq!(result.extract::<&str>().unwrap(), "called with args");
        // let np = PyModule::import(py, "numpy").unwrap();
        // let shape = np.getattr("shape").unwrap().  call1::<u16>(value).unwrap();
        let mut array: darrayf64 = ArrayD::zeros(IxDyn(&[2, 3]));
    
    })

}


