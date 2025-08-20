use pyo3::prelude::*;

mod common_tensor_exports;
pub mod ndarray_base;
pub mod tensor_error;
pub mod wgpu_base;

// delete in the aftermath
mod modifier;

#[cfg(feature = "wgpu")]
#[pymodule]
pub mod wgpu_tensor {

    #[pymodule_export]
    use super::wgpu_base::TensorPy;

    #[pymodule_export]
    use super::common_tensor_exports::Distribution;
}

#[cfg(feature = "ndarray")]
#[pymodule]
pub mod ndarray_tensor {

    #[pymodule_export]
    use super::ndarray_base::TensorPy;

    #[pymodule_export]
    use super::common_tensor_exports::Distribution;
}
