#![allow(unused)]
#![recursion_limit = "512"]

//! [`wrap-burn`] attempts to expose burn's modules and methods in a manner that permits it to work
//! as a python interface. This module exposes the [`burn::nn`] module.

use burn::nn::Linear;
use burn::nn::*;
use burn::prelude::*;
use pyo3::prelude::*;

// I thought send and Sync were implemented automatically??
macro_rules! implement_send_and_sync {
    ($name:ty) => {
        unsafe impl Send for $name {}
        unsafe impl Sync for $name {}
    };
}

macro_rules! implement_wgpu_interface {
    ($name:ident, $actual_type:ident) => {
        #[pyclass]
        pub struct $name {
            pub inner: $actual_type<Wgpu>,
        }
    };
}

macro_rules! for_normal_struct_enums {
    ($name:ident, $actual_type:ident) => {
        #[pyclass]
        pub struct $name(pub $actual_type);
    };
}

macro_rules! implement_ndarray_interface {
    ($name:ident, $actual_type:ty) => {
        #[pyclass]
        pub struct $name {
            pub inner: $actual_type<NdArray>,
        }
    };
}

#[cfg(feature = "wgpu")]
#[pymodule]
pub mod wgpu {

    use super::*;
    use burn::backend::wgpu::{Wgpu, WgpuDevice};

    #[pyclass]
    #[derive(Debug)]
    #[repr(transparent)]
    pub struct PyLinear {
        pub inner: Linear<Wgpu>,
    }

    #[pyclass]
    #[derive(Debug)]
    #[repr(transparent)]
    pub struct PyBatchNormConfig(BatchNormConfig);

    #[pyclass]
    #[repr(transparent)]
    pub struct PyBatchNormRecord {
        pub inner: BatchNormRecord<Wgpu, 1>,
    }

    #[pyclass]
    #[repr(transparent)]
    pub struct PyBiLSTM {
        pub inner: BiLstm<Wgpu>,
    }

    #[pyclass]
    pub struct PyBiLSTMConfig(pub BiLstmConfig);

    #[pyclass]
    #[derive(Debug)]
    #[repr(transparent)]
    pub struct PyDropout(pub Dropout);

    pub mod loss {
        use super::*;

        #[pyclass]
        pub struct PyBinaryCrossEntropy {
            pub inner: nn::loss::BinaryCrossEntropyLoss<Wgpu>,
        }

        #[pyclass]
        pub struct PyBinaryCrossEntropyConfig(pub nn::loss::BinaryCrossEntropyLossConfig);

        #[pyclass]
        pub struct PyCrossEntropyLoss {
            pub inner: nn::loss::CrossEntropyLoss<Wgpu>,
        }

        #[pyclass]
        pub struct PyHuberLoss(pub nn::loss::HuberLoss);

        #[pyclass]
        pub struct PyHuberLossConfig(pub nn::loss::HuberLossConfig);

        #[pyclass]
        pub struct MseLoss(pub nn::loss::MseLoss);

        #[pyclass]
        pub struct PoissonLoss(pub nn::loss::PoissonNllLoss);

        #[pyclass]
        pub struct PoissonLossConfig(pub nn::loss::PoissonNllLossConfig);

        implement_send_and_sync!(PyBinaryCrossEntropy);
        implement_send_and_sync!(PyCrossEntropyLoss);
    }

    implement_wgpu_interface!(PyGateController, GateController);
    implement_wgpu_interface!(PyEmbedding, Embedding);
    implement_wgpu_interface!(PyGroupNorm, GroupNorm);
    implement_wgpu_interface!(PyInstanceNorm, InstanceNorm);
    implement_wgpu_interface!(PyInstanceNormRecord, InstanceNormRecord);
    implement_wgpu_interface!(PyLayerNorm, LayerNorm);
    implement_wgpu_interface!(PyLayerNormRecord, LayerNormRecord);
    // implement_wgpu_interface!(PyLinearRecord, LinearRecord);
    implement_wgpu_interface!(PyLstm, Lstm);
    implement_wgpu_interface!(PyLstmRecord, LstmRecord);
    implement_wgpu_interface!(PyPRelu, PRelu);
    implement_wgpu_interface!(PyPReluRecord, PReluRecord);
    implement_wgpu_interface!(PyPositionalEncoding, PositionalEncoding);
    implement_wgpu_interface!(PyPositionalEncodingRecord, PositionalEncodingRecord);
    implement_wgpu_interface!(PyRmsNorm, RmsNorm);
    implement_wgpu_interface!(PyRmsNormRecord, RmsNormRecord);
    implement_wgpu_interface!(PyRotaryEncoding, RotaryEncoding);
    implement_wgpu_interface!(PyRotaryEncodingRecord, RotaryEncodingRecord);
    implement_wgpu_interface!(PySwiGlu, SwiGlu);
    // implement_wgpu_interface!(PySwiGluRecord, SwiGluRecord);

    for_normal_struct_enums!(PyUnfold4d, Unfold4d);
    for_normal_struct_enums!(PyUnfold4dConfig, Unfold4dConfig);
    for_normal_struct_enums!(PyTanh, Tanh);
    for_normal_struct_enums!(PySwiGluConfig, SwiGluConfig);
    for_normal_struct_enums!(PyPositionalEncodingConfig, PositionalEncodingConfig);
    for_normal_struct_enums!(PyPReluConfig, PReluConfig);
    for_normal_struct_enums!(PyLstmConfig, LstmConfig);
    for_normal_struct_enums!(PyLeakyRelu, LeakyRelu);
    for_normal_struct_enums!(PyLeakyReluConfig, LeakyReluConfig);
    for_normal_struct_enums!(PyGeLu, Gelu);
    for_normal_struct_enums!(PyHardSigmoid, HardSigmoid);
    for_normal_struct_enums!(PyHardSigmoidConfig, HardSigmoidConfig);
    for_normal_struct_enums!(PyInstanceNormConfig, InstanceNormConfig);
    for_normal_struct_enums!(PyLayerNormConfig, LayerNormConfig);
    for_normal_struct_enums!(PyRmsNormConfig, RmsNormConfig);
    for_normal_struct_enums!(PySigmoid, Sigmoid);
    for_normal_struct_enums!(PyInitializer, Initializer);
    for_normal_struct_enums!(PyPaddingConfig1d, PaddingConfig1d);
    for_normal_struct_enums!(PyPaddingConfig2d, PaddingConfig2d);
    for_normal_struct_enums!(PyPaddingConfig3d, PaddingConfig3d);

    implement_send_and_sync!(PySwiGlu);
    // implement_send_and_sync!(PySwiGluRecord);
    implement_send_and_sync!(PyRotaryEncoding);
    implement_send_and_sync!(PyRotaryEncodingRecord);
    implement_send_and_sync!(PyRmsNorm);
    implement_send_and_sync!(PyRmsNormRecord);
    implement_send_and_sync!(PyPositionalEncodingRecord);
    implement_send_and_sync!(PyPositionalEncoding);
    implement_send_and_sync!(PyPReluRecord);
    implement_send_and_sync!(PyPRelu);
    implement_send_and_sync!(PyLstm);
    implement_send_and_sync!(PyLstmRecord);
    // implement_send_and_sync!(PyLinearRecord);
    implement_send_and_sync!(PyLayerNorm);
    implement_send_and_sync!(PyLayerNormRecord);
    implement_send_and_sync!(PyInstanceNormRecord);
    implement_send_and_sync!(PyInstanceNorm);
    implement_send_and_sync!(PyEmbedding);
    implement_send_and_sync!(PyGroupNorm);
    implement_send_and_sync!(PyLinear);
    implement_send_and_sync!(PyBatchNormRecord);
    implement_send_and_sync!(PyBiLSTM);
    implement_send_and_sync!(PyGateController);

    #[pymodule]
    pub mod attention {
        use burn::nn::attention::*;
        use pyo3::prelude::*;
        use burn::prelude::*;
        use super::*;

        // vec![GeneratePaddingMask, MhaCache, MhaInput, MultiHeadAttention];

        implement_wgpu_interface!(PyGeneratePaddingMask,GeneratePaddingMask);
        implement_wgpu_interface!(PyMhaCache, MhaCache);
        implement_wgpu_interface!(PyMhaInput, MhaInput);
        implement_wgpu_interface!(PyMultiHeadAttention, MultiHeadAttention);
        implement_wgpu_interface!(PyMhaOutput, MhaOutput);
        implement_wgpu_interface!(PyMultiHeadAttentionRecord, MultiHeadAttentionRecord);

        for_normal_struct_enums!(PyMultiHeadAttentionConfig, MultiHeadAttentionConfig);

        implement_send_and_sync!(PyMultiHeadAttentionRecord);
        implement_send_and_sync!(PyMultiHeadAttention);
        implement_send_and_sync!(PyMhaOutput);
        
    }

    #[pymodule]
    pub mod conv {
        use super::*;
        use burn::prelude::*;
        use burn::nn::conv::*;

        implement_wgpu_interface!(PyDeformConv2d, DeformConv2d);
        implement_wgpu_interface!(PyDeformConv2dRecord, DeformConv2dRecord);
        implement_wgpu_interface!(PyConv1d, Conv1d);
        implement_wgpu_interface!(PyConv1dRecord, Conv1dRecord);
        implement_wgpu_interface!(PyConv2d, Conv2d);
        implement_wgpu_interface!(PyConv2dRecord, Conv2dRecord);
        implement_wgpu_interface!(PyConvTranspose1d, ConvTranspose1d);
        implement_wgpu_interface!(PyConvTranspose1dRecord, ConvTranspose1dRecord);
        implement_wgpu_interface!(PyConvTranspose2d, ConvTranspose2d);
        implement_wgpu_interface!(PyConvTranspose2dRecord, ConvTranspose2dRecord);
        implement_wgpu_interface!(PyConvTranspose3d, ConvTranspose3d);
        implement_wgpu_interface!(PyConvTranspose3dRecord, ConvTranspose3dRecord);

        for_normal_struct_enums!(PyConvTranspose1dConfig, ConvTranspose1dConfig);
        for_normal_struct_enums!(PyConvTranspose2dConfig, ConvTranspose2dConfig);
        for_normal_struct_enums!(PyConvTranspose3dConfig, ConvTranspose3dConfig);
        for_normal_struct_enums!(PyConv1DConfig, Conv1dConfig);
        for_normal_struct_enums!(PyConv2DConfig, Conv2dConfig);
        for_normal_struct_enums!(PyConv3DConfig, Conv3dConfig);

        implement_send_and_sync!(PyConv1d);
        implement_send_and_sync!(PyConv1dRecord);
        implement_send_and_sync!(PyConv2d);
        implement_send_and_sync!(PyConv2dRecord);
        implement_send_and_sync!(PyConvTranspose1d);
        implement_send_and_sync!(PyConvTranspose1dRecord);
        implement_send_and_sync!(PyConvTranspose2d);
        implement_send_and_sync!(PyConvTranspose2dRecord);
        implement_send_and_sync!(PyConvTranspose3d);
        implement_send_and_sync!(PyConvTranspose3dRecord);
        implement_send_and_sync!(PyDeformConv2d);
        implement_send_and_sync!(PyDeformConv2dRecord);
    }


}

#[cfg(feature = "ndarray")]
#[pymodule]
pub mod ndarray {

    use super::*;
    use burn::backend::ndarray::{NdArray, NdArrayDevice};

    #[pyclass]
    #[repr(transparent)]
    #[derive(Debug)]
    struct PyLinear {
        inner: Linear<NdArray>,
    }
    implement_send_and_sync!(PyLinear);
}

// [`TODO`] Item types unimmplemented
// [`TODO`] Implement configuration methods as python functions
