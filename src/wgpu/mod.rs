use pyo3::prelude::*;
mod nn;
mod optim;
mod tensor;
mod train;

/// This is the primary entry point for the wgpu backend
#[cfg(feature = "wgpu")]
#[pymodule]
pub mod wgpu {
    // use super::*;
    #[pymodule_export]
    use super::nn::mod_nn::nn;
    #[pymodule_export]
    use crate::module::package::wgpu_module::module;
}
