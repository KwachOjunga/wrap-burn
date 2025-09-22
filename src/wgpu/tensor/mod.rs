mod wgpu_base;
use pyo3::prelude::*;

#[pymodule]
pub mod wg_tensor {

    use super::*;
    #[pymodule]
    pub mod tensor {

        #[pymodule_export]
        use super::wgpu_base::TensorPy;

        #[pymodule_export]
        use crate::tensor::common_tensor_exports::Distribution;
    }
}
