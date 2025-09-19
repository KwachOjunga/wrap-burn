use pyo3::prelude::*;

mod optim;
mod tensor;
mod train;

/// This is the primary entry point for the wgpu backend
#[cfg(feature = "ndarray")]
#[pymodule]
mod ndarray {
    #[pymodule_export]
    use crate::module::package::nd_module::module;
}
