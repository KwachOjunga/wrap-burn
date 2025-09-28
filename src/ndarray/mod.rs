use pyo3::prelude::*;

mod nn;
mod tensor;
mod train;

/// This is the primary entry point for the wgpu backend
#[cfg(feature = "ndarray")]
#[pymodule]
pub mod ndarray {
    // use super::*;

    #[pymodule_export]
    use super::nn::nn;
    #[pymodule_export]
    use super::tensor::nd_tensor::tensor;
    #[pymodule_export]
    use super::train::ndarray_train::train;
    #[pymodule_export]
    use crate::module::package::nd_module::module;
    #[pymodule_export]
    use crate::optim::ndarray_optim::optim;
    #[pymodule_export]
    use crate::lr_scheduler::scheduler;

}
