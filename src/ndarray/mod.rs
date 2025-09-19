use pyo3::prelude::*;

mod nn;
mod optim;
mod tensor;
mod train;

/// This is the primary entry point for the wgpu backend
#[cfg(feature = "ndarray")]
#[pymodule]
pub mod ndarray {
    #[pymodule_export]
    use super::nn::nn;
    #[pymodule_export]
    use crate::module::package::nd_module::module;
}
