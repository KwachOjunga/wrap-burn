use burn::grad_clipping::*;
use pyo3::prelude::*;

#[pyclass]
pub struct GradientClippingPy(GradientClipping);

impl From<GradientClipping> for GradientClippingPy {
    fn from(other: GradientClipping) -> Self {
        Self(other)
    }
}

#[pymethods]
impl GradientClippingPy {
    #[new]
    fn new(val: Option<f32>, norm: Option<f32>) -> Self {
        let config = match (val, norm) {
            (Some(v), None) => GradientClippingConfig::Value(v),
            (None, Some(n)) => GradientClippingConfig::Norm(n),
            _ => panic!("Either 'val' or 'norm' must be provided, but not both."),
        };
        config.init().into()
    }

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
