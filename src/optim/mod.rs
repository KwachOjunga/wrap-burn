mod common_exports;



use pyo3::prelude::*;

#[cfg(feature = "wgpu")]
#[pymodule]
pub mod common {
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
    #[pymodule_export]
    use super::common_exports::RmsPropConfigPy;
    #[pymodule_export]
    use super::common_exports::RmsPropMomentumPy;
    #[pymodule_export]
    use super::common_exports::RmsPropPy;
    #[pymodule_export]
    use super::common_exports::SgdConfigPy;

    #[pymodule]
    pub mod decay {
        #[pymodule_export]
        use super::common_exports::decay_exports::WeightDecayConfigPy;
        #[pymodule_export]
        use super::common_exports::decay_exports::WeightDecayPy;
    }
}

#[cfg(feature = "ndarray")]
#[pymodule]
pub mod ndarray_optim {
    #[pymodule_export]
    use super::common;
}

#[cfg(feature = "wgpu")]
#[pymodule]
pub mod wgpu_optim {
    #[pymodule_export]
    use super::common;
}