#![allow(unused)]
#![recursion_limit = "512"]

//! [`wrap-burn`] attempts to expose burn's modules and methods in a manner that permits it to work
//! as a python interface. This module exposes the [`burn::nn`] module.

use crate::{
    for_normal_struct_enums, implement_ndarray_interface, implement_send_and_sync,
    implement_wgpu_interface,
};
use burn::nn::Linear;
use burn::nn::*;
use burn::prelude::*;
use pyo3::prelude::*;
mod wgpu_nn_exports;
mod ndarray_nn_exports;
// I thought send and Sync were implemented automatically??

/// Neural network Module as implemented using a WGPU backend
/// The module offers the typical building blocks relevant for
/// building elaborate `nn` architectures.
/// Includes; a conv module
///           - attention module -- for building transformer architectures
///           - cache module -- exposes the TensorCache
///           - gru module for the `Gated Recurrent Unit`
///           - loss module -- the loss functions
///           - lstm module --
///           - pool module -- exposing pooling layers particularly in use in CNN architectures
///           - transformer module
/// Some of these modules classes are re-exported at the base of the module
#[cfg(feature = "wgpu")]
#[pymodule]
pub mod wgpu_nn {

    use super::*;
    use burn::backend::wgpu::{Wgpu, WgpuDevice};

    #[pymodule_export]
    use wgpu_nn_exports::PyEmbedding;
    #[pymodule_export]
    use wgpu_nn_exports::PyGateController;
    #[pymodule_export]
    use wgpu_nn_exports::PyGeLu;
    #[pymodule_export]
    use wgpu_nn_exports::PyGroupNorm;
    #[pymodule_export]
    use wgpu_nn_exports::PyHardSigmoid;
    #[pymodule_export]
    use wgpu_nn_exports::PyInitializer;
    #[pymodule_export]
    use wgpu_nn_exports::PyInstanceNorm;
    #[pymodule_export]
    use wgpu_nn_exports::PyInstanceNormConfig;
    #[pymodule_export]
    use wgpu_nn_exports::PyInstanceNormRecord;
    #[pymodule_export]
    use wgpu_nn_exports::PyLeakyRelu;
    #[pymodule_export]
    use wgpu_nn_exports::PyLeakyReluConfig;
    #[pymodule_export]
    use wgpu_nn_exports::PyLstm;
    #[pymodule_export]
    use wgpu_nn_exports::PyLstmRecord;
    #[pymodule_export]
    use wgpu_nn_exports::PyPRelu;
    #[pymodule_export]
    use wgpu_nn_exports::PyLstmConfig;
    #[pymodule_export]
    use wgpu_nn_exports::PyPReluRecord;
    #[pymodule_export]
    use wgpu_nn_exports::PyPaddingConfig1d;
    #[pymodule_export]
    use wgpu_nn_exports::PyPaddingConfig2d;
    #[pymodule_export]
    use wgpu_nn_exports::PyPaddingConfig3d;
    #[pymodule_export]
    use wgpu_nn_exports::PyPositionalEncoding;
    #[pymodule_export]
    use wgpu_nn_exports::PyRmsNorm;
    #[pymodule_export]
    use wgpu_nn_exports::PyRmsNormConfig;
    #[pymodule_export]
    use wgpu_nn_exports::PyRmsNormRecord;
    #[pymodule_export]
    use wgpu_nn_exports::PyRotaryEncoding;
    #[pymodule_export]
    use wgpu_nn_exports::PyRotaryEncodingRecord;
    #[pymodule_export]
    use wgpu_nn_exports::PySigmoid;
    #[pymodule_export]
    use wgpu_nn_exports::PySwiGlu;
    #[pymodule_export]
    use wgpu_nn_exports::PySwiGluConfig;
    #[pymodule_export]
    use wgpu_nn_exports::PyTanh;
    #[pymodule_export]
    use wgpu_nn_exports::PyUnfold4d;
    #[pymodule_export]
    use wgpu_nn_exports::PyUnfold4dConfig;
    
    /// Applies Linear transformation over a tensor
    #[pyclass]
    #[derive(Debug)]
    #[repr(transparent)]
    pub struct PyLinear {
        pub inner: Linear<Wgpu>,
    }

    /// Offers an avenue to configure the BatchNorm layer
    #[pyclass]
    #[derive(Debug)]
    #[repr(transparent)]
    pub struct PyBatchNormConfig(BatchNormConfig);

    //[`TODO`] @kwach this `BatchNormRecord` is generic with two arguments; @kwach FIX this
    /// The record type for the BatchNorm module
    #[pyclass]
    #[repr(transparent)]
    pub struct PyBatchNormRecord {
        pub inner: BatchNormRecord<Wgpu, 1>,
    }

    /// The implementation of the Bidirectional LSTM module.
    #[pyclass]
    #[repr(transparent)]
    pub struct PyBiLSTM {
        pub inner: BiLstm<Wgpu>,
    }

    /// Configuraation to build the BiLSTM module
    #[pyclass]
    pub struct PyBiLSTMConfig(pub BiLstmConfig);

    /// The Dropout layer; set at random elements of the input tensor to zero during training.
    #[pyclass]
    #[derive(Debug)]
    #[repr(transparent)]
    pub struct PyDropout(pub Dropout);

    implement_send_and_sync!(PyLinear);
    implement_send_and_sync!(PyBatchNormRecord);
    implement_send_and_sync!(PyBiLSTM);
    
    /// Loss module that exposes various loss functions
    #[pymodule]
    pub mod loss {
        use super::*;

        /// The BinaryCrossEntropyLoss; calculate oss from input logits and targets
        #[pyclass]
        pub struct PyBinaryCrossEntropy {
            pub inner: nn::loss::BinaryCrossEntropyLoss<Wgpu>,
        }

        /// Configuration to build the BinaryCrossEntropyLoss
        #[pyclass]
        pub struct PyBinaryCrossEntropyConfig(pub nn::loss::BinaryCrossEntropyLossConfig);

        /// calculate cross entropy loss from input logits to target
        #[pyclass]
        pub struct PyCrossEntropyLoss {
            pub inner: nn::loss::CrossEntropyLoss<Wgpu>,
        }

        /// Calculate the HuberLoss between inputs and target
        #[pyclass]
        pub struct PyHuberLoss(pub nn::loss::HuberLoss);

        /// Configuration to build the HuberLoss
        #[pyclass]
        pub struct PyHuberLossConfig(pub nn::loss::HuberLossConfig);

        /// Calculate the mean squared error loss from the input logits and the targets.
        #[pyclass]
        pub struct MseLoss(pub nn::loss::MseLoss);

        /// Negative Log Likelihood (NLL) loss with a Poisson distribution assumption for the target.
        #[pyclass]
        pub struct PoissonLoss(pub nn::loss::PoissonNllLoss);

        /// Configuration to calculate the PoissonLoss
        #[pyclass]
        pub struct PoissonLossConfig(pub nn::loss::PoissonNllLossConfig);

        implement_send_and_sync!(PyBinaryCrossEntropy);
        implement_send_and_sync!(PyCrossEntropyLoss);
    }

    #[pymodule]
    pub mod attention {
        use super::*;
        use burn::nn::attention::*;
        use burn::prelude::*;
        use pyo3::prelude::*;

        // vec![GeneratePaddingMask, MhaCache, MhaInput, MultiHeadAttention];

        implement_wgpu_interface!(PyGeneratePaddingMask, GeneratePaddingMask);
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
        use burn::nn::conv::*;
        use burn::prelude::*;

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

    #[pymodule]
    pub mod gru {
        use super::*;
        use burn::nn::gru::*;

        implement_wgpu_interface!(PyGru, Gru);
        implement_wgpu_interface!(PyGruRecord, GruRecord);

        for_normal_struct_enums!(PyGruConfig, GruConfig);

        implement_send_and_sync!(PyGruRecord);
        implement_send_and_sync!(PyGru);
    }

    #[pymodule]
    pub mod interpolate {
        use super::*;
        use burn::nn::interpolate::*;
        use burn::prelude::*;
        use pyo3::prelude::*;

        for_normal_struct_enums!(PyInterpolate1d, Interpolate1d);
        for_normal_struct_enums!(PyInterpolate1dConfig, Interpolate1dConfig);
        for_normal_struct_enums!(PyInterpolate2d, Interpolate2d);
        for_normal_struct_enums!(PyInterpolate2dConfig, Interpolate2dConfig);
        for_normal_struct_enums!(PyInterpolateMode, InterpolateMode);
    }

    mod pool_exports {
        pub(crate) use super::*;
        use burn::nn::pool::*;

        /// This is  the typical AdaptivePool1d layer
        for_normal_struct_enums!(PyAdaptiveAvgPool1d, AdaptiveAvgPool1d);
        for_normal_struct_enums!(PyAdaptiveAvgPool1dConfig, AdaptiveAvgPool1dConfig);
        for_normal_struct_enums!(PyAdaptiveAvgPool2d, AdaptiveAvgPool2d);
        for_normal_struct_enums!(PyAdaptiveAvgPool2dConfig, AdaptiveAvgPool2dConfig);
        for_normal_struct_enums!(PyAvgPool1d, AvgPool1d);
        for_normal_struct_enums!(PyAvgPool1dConfig, AvgPool1dConfig);
        for_normal_struct_enums!(PyAvgPool2d, AvgPool2d);
        for_normal_struct_enums!(PyAvgPool2dConfig, AvgPool2dConfig);
        for_normal_struct_enums!(PyMaxPool1d, MaxPool1d);
        for_normal_struct_enums!(PyMaxPool1dConfig, MaxPool1dConfig);
        for_normal_struct_enums!(PyMaxPool2d, MaxPool2d);
        for_normal_struct_enums!(PyMaxPool2dConfig, MaxPool2dConfig);
    }

    #[pymodule]
    pub mod pool {
        use super::*;

        #[pymodule_export]
        use super::pool_exports::PyAdaptiveAvgPool1d;

        #[pymodule_export]
        use super::pool_exports::PyAdaptiveAvgPool1dConfig;

        #[pymodule_export]
        use super::pool_exports::PyAdaptiveAvgPool2d;

        #[pymodule_export]
        use super::pool_exports::PyAdaptiveAvgPool2dConfig;

        #[pymodule_export]
        use super::pool_exports::PyAvgPool1d;
    }

    #[pymodule]
    pub mod transformer {
        use super::*;
        use burn::nn::transformer::*;

        #[pyclass]
        pub struct PyPositionWiseFeedForward {
            inner: PositionWiseFeedForward<Wgpu>,
        }
        // implement_wgpu_interface!(PyPositionWiseFeedForward, PositionWiseFeedForward);

        implement_wgpu_interface!(
            PyPositionWiseFeedForwardRecord,
            PositionWiseFeedForwardRecord
        );

        implement_wgpu_interface!(PyTransformerDecoder, TransformerDecoder);
        implement_wgpu_interface!(
            PyTransformerDecoderAutoregressiveCache,
            TransformerDecoderAutoregressiveCache
        );
        implement_wgpu_interface!(PyTransformerDecoderInput, TransformerDecoderInput);
        implement_wgpu_interface!(PyTransformerDecoderLayer, TransformerDecoderLayer);
        implement_wgpu_interface!(
            PyTransformerDecoderLayerRecord,
            TransformerDecoderLayerRecord
        );
        implement_wgpu_interface!(PyTransformerDecoderRecord, TransformerDecoderRecord);
        implement_wgpu_interface!(PyTransformerEncoder, TransformerEncoder);
        implement_wgpu_interface!(
            PyTransformerEncoderAutoregressiveCache,
            TransformerEncoderAutoregressiveCache
        );
        implement_wgpu_interface!(PyTransformerEncoderLayer, TransformerEncoderLayer);
        implement_wgpu_interface!(
            PyTransformerEncoderLayerRecord,
            TransformerEncoderLayerRecord
        );
        implement_wgpu_interface!(PyTransformerEncoderRecord, TransformerEncoderRecord);
        implement_wgpu_interface!(PyTransformerEncoderInput, TransformerEncoderInput);

        for_normal_struct_enums!(
            PyPositionWiseFeedForwardConfig,
            PositionWiseFeedForwardConfig
        );
        for_normal_struct_enums!(PyTransformerDecoderConfig, TransformerDecoderConfig);

        implement_send_and_sync!(PyTransformerEncoderRecord);
        implement_send_and_sync!(PyTransformerEncoderLayerRecord);
        implement_send_and_sync!(PyTransformerEncoderLayer);
        implement_send_and_sync!(PyTransformerEncoderInput);
        implement_send_and_sync!(PyTransformerEncoderAutoregressiveCache);
        implement_send_and_sync!(PyTransformerEncoder);
        implement_send_and_sync!(PyTransformerDecoderRecord);
        implement_send_and_sync!(PyTransformerDecoderLayerRecord);
        implement_send_and_sync!(PyTransformerDecoderLayer);
        implement_send_and_sync!(PyTransformerDecoderInput);
        implement_send_and_sync!(PyTransformerDecoderAutoregressiveCache);
        implement_send_and_sync!(PyTransformerDecoder);
        implement_send_and_sync!(PyPositionWiseFeedForward);
        implement_send_and_sync!(PyPositionWiseFeedForwardRecord);
    }
}



/// Neural network Module as implemented using a NdArray backend
/// Basically, this means whatever training or inference will be perfomed
/// by the CPU.
/// The module offers the typical building blocks relevant for
/// building elaborate `nn` architectures.
/// Includes; a conv module
///           - attention module -- for building transformer architectures
///           - cache module -- exposes the TensorCache
///           - gru module for the `Gated Recurrent Unit`
///           - loss module -- the loss functions
///           - lstm module --
///           - pool module -- exposing pooling layers particularly in use in CNN architectures
///           - transformer module
/// Some of these modules classes are re-exported at the base of the module
#[cfg(feature = "ndarray")]
#[pymodule]
pub mod ndarray {

    use super::*;
    use burn::backend::ndarray::*;


    #[pymodule_export]
    use ndarray_nn_exports::PyEmbedding;
    #[pymodule_export]
    use ndarray_nn_exports::PyGateController;
    #[pymodule_export]
    use ndarray_nn_exports::PyGeLu;
    #[pymodule_export]
    use ndarray_nn_exports::PyGroupNorm;
    #[pymodule_export]
    use ndarray_nn_exports::PyHardSigmoid;
    #[pymodule_export]
    use ndarray_nn_exports::PyInitializer;
    #[pymodule_export]
    use ndarray_nn_exports::PyInstanceNorm;
    #[pymodule_export]
    use ndarray_nn_exports::PyInstanceNormConfig;
    #[pymodule_export]
    use ndarray_nn_exports::PyInstanceNormRecord;
    #[pymodule_export]
    use ndarray_nn_exports::PyLeakyRelu;
    #[pymodule_export]
    use ndarray_nn_exports::PyLeakyReluConfig;
    #[pymodule_export]
    use ndarray_nn_exports::PyLstm;
    #[pymodule_export]
    use ndarray_nn_exports::PyLstmRecord;
    #[pymodule_export]
    use ndarray_nn_exports::PyPRelu;
    #[pymodule_export]
    use ndarray_nn_exports::PyLstmConfig;
    #[pymodule_export]
    use ndarray_nn_exports::PyPReluRecord;
    #[pymodule_export]
    use ndarray_nn_exports::PyPaddingConfig1d;
    #[pymodule_export]
    use ndarray_nn_exports::PyPaddingConfig2d;
    #[pymodule_export]
    use ndarray_nn_exports::PyPaddingConfig3d;
    #[pymodule_export]
    use ndarray_nn_exports::PyPositionalEncoding;
    #[pymodule_export]
    use ndarray_nn_exports::PyRmsNorm;
    #[pymodule_export]
    use ndarray_nn_exports::PyRmsNormConfig;
    #[pymodule_export]
    use ndarray_nn_exports::PyRmsNormRecord;
    #[pymodule_export]
    use ndarray_nn_exports::PyRotaryEncoding;
    #[pymodule_export]
    use ndarray_nn_exports::PyRotaryEncodingRecord;
    #[pymodule_export]
    use ndarray_nn_exports::PySigmoid;
    #[pymodule_export]
    use ndarray_nn_exports::PySwiGlu;
    #[pymodule_export]
    use ndarray_nn_exports::PySwiGluConfig;
    #[pymodule_export]
    use ndarray_nn_exports::PyTanh;
    #[pymodule_export]
    use ndarray_nn_exports::PyUnfold4d;
    #[pymodule_export]
    use ndarray_nn_exports::PyUnfold4dConfig;
    
    
   
    #[pyclass]
    #[derive(Debug)]
    #[repr(transparent)]
    pub struct PyLinear {
        pub inner: Linear<NdArray>,
    }

    #[pyclass]
    #[derive(Debug)]
    #[repr(transparent)]
    pub struct PyBatchNormConfig(BatchNormConfig);

    #[pyclass]
    #[repr(transparent)]
    pub struct PyBatchNormRecord {
        pub inner: BatchNormRecord<NdArray, 1>,
    }

    #[pyclass]
    #[repr(transparent)]
    pub struct PyBiLSTM {
        pub inner: BiLstm<NdArray>,
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
            pub inner: nn::loss::BinaryCrossEntropyLoss<NdArray>,
        }

        #[pyclass]
        pub struct PyBinaryCrossEntropyConfig(pub nn::loss::BinaryCrossEntropyLossConfig);

        #[pyclass]
        pub struct PyCrossEntropyLoss {
            pub inner: nn::loss::CrossEntropyLoss<NdArray>,
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

   
    implement_send_and_sync!(PyLinear);
    implement_send_and_sync!(PyBatchNormRecord);
    implement_send_and_sync!(PyBiLSTM);
   

    #[pymodule]
    pub mod attention {
        use super::*;
        use burn::nn::attention::*;
        use burn::prelude::*;
        use pyo3::prelude::*;

        // vec![GeneratePaddingMask, MhaCache, MhaInput, MultiHeadAttention];

        implement_ndarray_interface!(PyGeneratePaddingMask, GeneratePaddingMask);
        implement_ndarray_interface!(PyMhaCache, MhaCache);
        implement_ndarray_interface!(PyMhaInput, MhaInput);
        implement_ndarray_interface!(PyMultiHeadAttention, MultiHeadAttention);
        implement_ndarray_interface!(PyMhaOutput, MhaOutput);
        implement_ndarray_interface!(PyMultiHeadAttentionRecord, MultiHeadAttentionRecord);

        for_normal_struct_enums!(PyMultiHeadAttentionConfig, MultiHeadAttentionConfig);

        implement_send_and_sync!(PyMultiHeadAttentionRecord);
        implement_send_and_sync!(PyMultiHeadAttention);
        implement_send_and_sync!(PyMhaOutput);
    }

    #[pymodule]
    pub mod conv {
        use super::*;
        use burn::nn::conv::*;
        use burn::prelude::*;

        implement_ndarray_interface!(PyDeformConv2d, DeformConv2d);
        implement_ndarray_interface!(PyDeformConv2dRecord, DeformConv2dRecord);
        implement_ndarray_interface!(PyConv1d, Conv1d);
        implement_ndarray_interface!(PyConv1dRecord, Conv1dRecord);
        implement_ndarray_interface!(PyConv2d, Conv2d);
        implement_ndarray_interface!(PyConv2dRecord, Conv2dRecord);
        implement_ndarray_interface!(PyConvTranspose1d, ConvTranspose1d);
        implement_ndarray_interface!(PyConvTranspose1dRecord, ConvTranspose1dRecord);
        implement_ndarray_interface!(PyConvTranspose2d, ConvTranspose2d);
        implement_ndarray_interface!(PyConvTranspose2dRecord, ConvTranspose2dRecord);
        implement_ndarray_interface!(PyConvTranspose3d, ConvTranspose3d);
        implement_ndarray_interface!(PyConvTranspose3dRecord, ConvTranspose3dRecord);

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

    #[pymodule]
    pub mod gru {
        use super::*;
        use burn::nn::gru::*;

        implement_ndarray_interface!(PyGru, Gru);
        implement_ndarray_interface!(PyGruRecord, GruRecord);

        for_normal_struct_enums!(PyGruConfig, GruConfig);

        implement_send_and_sync!(PyGruRecord);
        implement_send_and_sync!(PyGru);
    }

    #[pymodule]
    pub mod interpolate {
        use super::*;
        use burn::nn::interpolate::*;
        use burn::prelude::*;
        use pyo3::prelude::*;

        for_normal_struct_enums!(PyInterpolate1d, Interpolate1d);
        for_normal_struct_enums!(PyInterpolate1dConfig, Interpolate1dConfig);
        for_normal_struct_enums!(PyInterpolate2d, Interpolate2d);
        for_normal_struct_enums!(PyInterpolate2dConfig, Interpolate2dConfig);
        for_normal_struct_enums!(PyInterpolateMode, InterpolateMode);
    }

    #[pymodule]
    pub mod pool {
        use super::*;
        use burn::nn::pool::*;

        for_normal_struct_enums!(PyAdaptiveAvgPool1d, AdaptiveAvgPool1d);
        for_normal_struct_enums!(PyAdaptiveAvgPool1dConfig, AdaptiveAvgPool1dConfig);
        for_normal_struct_enums!(PyAdaptiveAvgPool2d, AdaptiveAvgPool2d);
        for_normal_struct_enums!(PyAdaptiveAvgPool2dConfig, AdaptiveAvgPool2dConfig);
        for_normal_struct_enums!(PyAvgPool1d, AvgPool1d);
        for_normal_struct_enums!(PyAvgPool1dConfig, AvgPool1dConfig);
        for_normal_struct_enums!(PyAvgPool2d, AvgPool2d);
        for_normal_struct_enums!(PyAvgPool2dConfig, AvgPool2dConfig);
        for_normal_struct_enums!(PyMaxPool1d, MaxPool1d);
        for_normal_struct_enums!(PyMaxPool1dConfig, MaxPool1dConfig);
        for_normal_struct_enums!(PyMaxPool2d, MaxPool2d);
        for_normal_struct_enums!(PyMaxPool2dConfig, MaxPool2dConfig);
    }

    #[pymodule]
    pub mod transformer {
        use super::*;
        use burn::nn::transformer::*;

        implement_ndarray_interface!(PyPositionWiseFeedForward, PositionWiseFeedForward);
        implement_ndarray_interface!(
            PyPositionWiseFeedForwardRecord,
            PositionWiseFeedForwardRecord
        );
        implement_ndarray_interface!(PyTransformerDecoder, TransformerDecoder);
        implement_ndarray_interface!(
            PyTransformerDecoderAutoregressiveCache,
            TransformerDecoderAutoregressiveCache
        );
        implement_ndarray_interface!(PyTransformerDecoderInput, TransformerDecoderInput);
        implement_ndarray_interface!(PyTransformerDecoderLayer, TransformerDecoderLayer);
        implement_ndarray_interface!(
            PyTransformerDecoderLayerRecord,
            TransformerDecoderLayerRecord
        );
        implement_ndarray_interface!(PyTransformerDecoderRecord, TransformerDecoderRecord);
        implement_ndarray_interface!(PyTransformerEncoder, TransformerEncoder);
        implement_ndarray_interface!(
            PyTransformerEncoderAutoregressiveCache,
            TransformerEncoderAutoregressiveCache
        );
        implement_ndarray_interface!(PyTransformerEncoderLayer, TransformerEncoderLayer);
        implement_ndarray_interface!(
            PyTransformerEncoderLayerRecord,
            TransformerEncoderLayerRecord
        );
        implement_ndarray_interface!(PyTransformerEncoderRecord, TransformerEncoderRecord);
        implement_ndarray_interface!(PyTransformerEncoderInput, TransformerEncoderInput);

        for_normal_struct_enums!(
            PyPositionWiseFeedForwardConfig,
            PositionWiseFeedForwardConfig
        );
        for_normal_struct_enums!(PyTransformerDecoderConfig, TransformerDecoderConfig);

        implement_send_and_sync!(PyTransformerEncoderRecord);
        implement_send_and_sync!(PyTransformerEncoderLayerRecord);
        implement_send_and_sync!(PyTransformerEncoderLayer);
        implement_send_and_sync!(PyTransformerEncoderInput);
        implement_send_and_sync!(PyTransformerEncoderAutoregressiveCache);
        implement_send_and_sync!(PyTransformerEncoder);
        implement_send_and_sync!(PyTransformerDecoderRecord);
        implement_send_and_sync!(PyTransformerDecoderLayerRecord);
        implement_send_and_sync!(PyTransformerDecoderLayer);
        implement_send_and_sync!(PyTransformerDecoderInput);
        implement_send_and_sync!(PyTransformerDecoderAutoregressiveCache);
        implement_send_and_sync!(PyTransformerDecoder);
        implement_send_and_sync!(PyPositionWiseFeedForward);
        implement_send_and_sync!(PyPositionWiseFeedForwardRecord);
    }
}

// [`TODO`] Item types unimmplemented
// [`TODO`] Implement configuration methods as python functions
