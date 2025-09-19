use pyo3::prelude::*;
use crate::common;

mod module;
mod tensor;
mod train;
mod optim;

/// This is the primary entry point for the wgpu backend
#[cfg(feature = "ndarray")]
#[pymodule]
mod ndarray {

}