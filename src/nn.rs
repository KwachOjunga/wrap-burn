#![allow(unused)]

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
}

#[cfg(feature = "ndarray")]
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
