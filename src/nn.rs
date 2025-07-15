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
        pub struct PyHuberLoss (pub nn::loss::HuberLoss);

        #[pyclass]
        pub struct PyHuberLossConfig(pub nn::loss::HuberLossConfig);

        #[pyclass]
        pub struct MseLoss (pub nn::loss::MseLoss);

        #[pyclass]
        pub struct PoissonLoss (pub nn::loss::PoissonNllLoss);

        #[pyclass]
        pub struct PoissonLossConfig (pub nn::loss::PoissonNllLossConfig);

        implement_send_and_sync!(PyBinaryCrossEntropy);
        implement_send_and_sync!(PyCrossEntropyLoss);
    }


    implement_send_and_sync!(PyLinear);
    implement_send_and_sync!(PyBatchNormRecord);
    implement_send_and_sync!(PyBiLSTM);
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
