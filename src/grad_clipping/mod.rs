use burn::grad_clipping::*;
use pyo3::prelude::*;
use pyo3::types::PyFloat;

#[pyclass]
pub struct GradientClippingPy (GradientClipping);

impl From<GradientClipping> for GradientClippingPy {
    fn from(other: GradientClipping) -> Self {
        Self(other)
    }
}


#[pymethods]
impl GradientClippingPy {

    #[staticmethod]
    fn by_value(val: f32) -> Self {

//        let val: f32 = val.extract().unwrap();
        GradientClippingConfig::Value(val).init().into()
    }


    #[staticmethod]
    fn by_norm(norm: f32) -> Self {
  //      let norm: f32 = norm.extract().unwrap();
        GradientClippingConfig::Norm(norm).init().into()
    }
}
