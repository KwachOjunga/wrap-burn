use pyo3::prelude::*;

mod ndarray_base;

#[pymodule]
mod nd_tensor {

    use super::*;
    #[pymodule]
    mod tensor {
        #[pymodule_export]
        use super::ndarray_base::TensorPy;

        #[pymodule_export]
        use crate::tensor::common_tensor_exports::Distribution;
    }
}
