use burn::grad_clipping::*;
use pyo3::prelude::*;


#[pyclass]
pub struct GradientClippingPy (GradientClipping);

impl From<GradientClipping> for GradientClippingPy {
    fn from(other: GradientClipping) -> Self {
        Self(other)
    }
}


#[pymethods]
impl GradientClippingPy {
    #[new]
    fn new() {
        GradientClippingConfig::new().into()
    }
}
