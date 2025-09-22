use pyo3::prelude::*;
pub mod nn;
pub mod tensor;
pub mod train;

/// This is the primary entry point for the wgpu backend
#[cfg(feature = "wgpu")]
#[pymodule]
pub mod wgpu {
    // use super::*;
    #[pymodule_export]
    use super::nn::mod_nn::nn;
    #[pymodule_export]
    use super::tensor::wg_tensor::tensor;
    #[pymodule_export]
    use super::train::wgpu_train::train;
    #[pymodule_export]
    use crate::module::package::wgpu_module::module;
    #[pymodule_export]
    use crate::optim::wgpu_optim::optim;
}
