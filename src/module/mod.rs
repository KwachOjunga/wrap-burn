//! This implements burn::module
//!
//! Some of the structs implemented here are necessary for
//! operations like quantization. This enables a user to opt in
//! to such features if they wish`to.
//!
//! Note: This is untested code, so it may not work as expected.

// [TODO]: Implement this crate's function's in a capacity that illustrates its utility.

use crate::for_normal_struct_enums;
use burn::backend::{NdArray, Wgpu};
use burn::module::*;
use burn::prelude::*;
use pyo3::prelude::*;

for_normal_struct_enums!(AttributePy, Attribute);
for_normal_struct_enums!(ConstantRecordPy, ConstantRecord);
for_normal_struct_enums!(ContentPy, Content);
for_normal_struct_enums!(DisplaySettingsPy, DisplaySettings);

for_normal_struct_enums!(QuantizerPy, Quantizer);

// Creating an avenue for constructing default neural network modules from python

#[derive(Module, Debug)]
struct NNModule<B: Backend> {
    // This is a placeholder for future fields
    _marker: std::marker::PhantomData<B>,
}

#[pymodule]
pub mod module {
    use super::*;

    #[cfg(feature = "wgpu")]
    #[pymodule]
    pub mod wgpu_module {
        use super::*;

        #[pyclass]
        pub struct WgpuNNModulePy {
            inner: NNModule<Wgpu>,
        }

        impl From<NNModule<Wgpu>> for WgpuNNModulePy {
            fn from(other: NNModule<Wgpu>) -> Self {
                Self { inner: other }
            }
        }

        #[pymethods]
        impl WgpuNNModulePy {
            fn quantize_weights(&self, quant: &mut QuantizerPy) -> Self {
                self.inner.clone().quantize_weights(&mut quant.0).into()
            }
        }
    }

    #[cfg(feature = "ndarray")]
    #[pymodule]
    pub mod nd_module {
        use super::*;

        #[pyclass]
        pub struct NdArrayNModulePy {
            inner: NNModule<NdArray>,
        }

        impl From<NNModule<NdArray>> for NdArrayNModulePy {
            fn from(other: NNModule<NdArray>) -> Self {
                Self { inner: other }
            }
        }

        #[pymethods]
        impl NdArrayNModulePy {
            fn quantize_weights(&self, quant: &mut QuantizerPy) -> Self {
                self.inner.clone().quantize_weights(&mut quant.0).into()
            }
        }
    }
}
