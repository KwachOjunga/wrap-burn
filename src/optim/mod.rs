#![allow(unused)]

mod common_exports;
mod ndarray_optim_exports;
mod wgpu_optim_exports;

use burn::optim::*;
use burn::prelude::*;
use pyo3::prelude::*;

#[cfg(feature = "wgpu")]
#[pymodule]
pub mod wgpu_optim {
    use super::*;

    #[pymodule_export]
    use super::common_exports::AdaGradConfigPy;
    #[pymodule_export]
    use super::common_exports::AdaGradPy;
    #[pymodule_export]
    use super::common_exports::AdamConfigPy;
    #[pymodule_export]
    use super::common_exports::AdamPy;
    #[pymodule_export]
    use super::common_exports::AdamWConfigPy;
    #[pymodule_export]
    use super::common_exports::AdamWPy;
    #[pymodule_export]
    use super::common_exports::GradientsParamsPy;
}

#[cfg(feature = "ndarray")]
pub mod ndarray {}
