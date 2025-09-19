use  pyo3::prelude::*;
use crate :: common;
mod tensor;
mod train;
mod module;
mod optim;

/// This is the primary entry point for the wgpu backend
#[cfg(feature = "wgpu")]
#[pymodule]
mod wgpu {
    use super::*;
    
}