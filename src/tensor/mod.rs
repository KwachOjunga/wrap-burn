use pyo3::prelude::*;

pub mod wgpu_base;
pub mod ndarray_base;
pub mod tensor_error;

// delete in the aftermath
mod modifier;

#[pymodule]
pub mod tensor {
    use super::*;

    #[cfg(feature = "wgpu")]
    #[pymodule]
    pub mod wgpu_tensor {
        
        #[pymodule_export]
        use super::wgpu_base::TensorPy;
    }

    #[cfg(feature = "ndarray")]
    #[pymodule]
    pub mod ndarray_tensor {
        
        #[pymodule_export]
        use super::ndarray_base::TensorPy;
    }
}