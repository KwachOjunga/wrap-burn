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

mod common_nn_exports;
mod ndarray_nn_exports;
mod wgpu_nn_exports;
// I thought send and Sync were implemented automatically??

/// Neural network Module as implemented using a WGPU backend
/// The module offers the typical building blocks relevant for
/// building elaborate `nn` architectures.
/// Includes; - conv module
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
    use wgpu_nn_exports::EmbeddingPy;
    #[pymodule_export]
    use wgpu_nn_exports::GateControllerPy;
    #[pymodule_export]
    use wgpu_nn_exports::GeLuPy;
    #[pymodule_export]
    use wgpu_nn_exports::GroupNormPy;
    #[pymodule_export]
    use wgpu_nn_exports::HardSigmoidPy;
    #[pymodule_export]
    use wgpu_nn_exports::InitializerPy;
    #[pymodule_export]
    use wgpu_nn_exports::InstanceNormConfigPy;
    #[pymodule_export]
    use wgpu_nn_exports::InstanceNormPy;
    #[pymodule_export]
    use wgpu_nn_exports::InstanceNormRecordPy;
    #[pymodule_export]
    use wgpu_nn_exports::LeakyReluConfigPy;
    #[pymodule_export]
    use wgpu_nn_exports::LeakyReluPy;
    #[pymodule_export]
    use wgpu_nn_exports::LstmConfigPy;
    #[pymodule_export]
    use wgpu_nn_exports::LstmPy;
    #[pymodule_export]
    use wgpu_nn_exports::LstmRecordPy;
    #[pymodule_export]
    use wgpu_nn_exports::PReluPy;
    #[pymodule_export]
    use wgpu_nn_exports::PReluRecordPy;
    #[pymodule_export]
    use wgpu_nn_exports::PaddingConfig1dPy;
    #[pymodule_export]
    use wgpu_nn_exports::PaddingConfig2dPy;
    #[pymodule_export]
    use wgpu_nn_exports::PaddingConfig3dPy;
    #[pymodule_export]
    use wgpu_nn_exports::PositionalEncodingPy;
    #[pymodule_export]
    use wgpu_nn_exports::RmsNormConfigPy;
    #[pymodule_export]
    use wgpu_nn_exports::RmsNormPy;
    #[pymodule_export]
    use wgpu_nn_exports::RmsNormRecordPy;
    #[pymodule_export]
    use wgpu_nn_exports::RotaryEncodingPy;
    #[pymodule_export]
    use wgpu_nn_exports::RotaryEncodingRecordPy;
    #[pymodule_export]
    use wgpu_nn_exports::SigmoidPy;
    #[pymodule_export]
    use wgpu_nn_exports::SwiGluConfigPy;
    #[pymodule_export]
    use wgpu_nn_exports::SwiGluPy;
    #[pymodule_export]
    use wgpu_nn_exports::TanhPy;

    // [TODO:] Note the current implementation of this
    #[pymodule_export]
    use crate::tensor::base::TensorPy;
    #[pymodule_export]
    use wgpu_nn_exports::Unfold4dConfigPy;
    #[pymodule_export]
    use wgpu_nn_exports::Unfold4dPy;

    /// Applies Linear transformation over a tensor
    #[pyclass]
    #[derive(Debug)]
    #[repr(transparent)]
    pub struct LinearPy {
        pub inner: Linear<Wgpu>,
    }

    /// Offers an avenue to configure the BatchNorm layer
    #[pyclass]
    #[derive(Debug)]
    #[repr(transparent)]
    pub struct BatchNormConfigPy(BatchNormConfig);

    //[`TODO`] @kwach this `BatchNormRecord` is generic with two arguments; @kwach FIX this
    /// The record type for the BatchNorm module
    #[pyclass]
    #[repr(transparent)]
    pub struct BatchNormRecordPy {
        pub inner: BatchNormRecord<Wgpu, 1>,
    }

    /// The implementation of the Bidirectional LSTM module.
    #[pyclass]
    #[repr(transparent)]
    pub struct BiLSTMPy {
        pub inner: BiLstm<Wgpu>,
    }

    /// Configuraation to build the BiLSTM module
    #[pyclass]
    pub struct BiLSTMConfigPy(pub BiLstmConfig);

    /// The Dropout layer; set at random elements of the input tensor to zero during training.
    #[pyclass]
    #[derive(Debug)]
    #[repr(transparent)]
    pub struct DropoutPy(pub Dropout);

    implement_send_and_sync!(LinearPy);
    implement_send_and_sync!(BatchNormRecordPy);
    implement_send_and_sync!(BiLSTMPy);

    /// Loss module that exposes various loss functions
    #[pymodule]
    pub mod loss {
        use super::*;

        /// The BinaryCrossEntropyLoss; calculate oss from input logits and targets
        #[pyclass]
        pub struct BinaryCrossEntropyPy {
            pub inner: nn::loss::BinaryCrossEntropyLoss<Wgpu>,
        }

        /// Configuration to build the BinaryCrossEntropyLoss
        #[pyclass]
        pub struct BinaryCrossEntropyConfigPy(pub nn::loss::BinaryCrossEntropyLossConfig);

        /// calculate cross entropy loss from input logits to target
        #[pyclass]
        pub struct CrossEntropyLossPy {
            pub inner: nn::loss::CrossEntropyLoss<Wgpu>,
        }

        /// Calculate the HuberLoss between inputs and target
        #[pyclass]
        pub struct HuberLossPy(pub nn::loss::HuberLoss);

        /// Configuration to build the HuberLoss
        #[pyclass]
        pub struct HuberLossConfigPy(pub nn::loss::HuberLossConfig);

        /// Calculate the mean squared error loss from the input logits and the targets.
        #[pyclass]
        pub struct MseLoss(pub nn::loss::MseLoss);

        /// Negative Log Likelihood (NLL) loss with a Poisson distribution assumption for the target.
        #[pyclass]
        pub struct PoissonLoss(pub nn::loss::PoissonNllLoss);

        /// Configuration to calculate the PoissonLoss
        #[pyclass]
        pub struct PoissonLossConfig(pub nn::loss::PoissonNllLossConfig);

        implement_send_and_sync!(BinaryCrossEntropyPy);
        implement_send_and_sync!(CrossEntropyLossPy);
    }

    #[pymodule]
    pub mod attention {
        use super::*;

        #[pymodule_export]
        use wgpu_nn_exports::attention_exports::GeneratePaddingMaskPy;
        #[pymodule_export]
        use wgpu_nn_exports::attention_exports::MhaCachePy;
        #[pymodule_export]
        use wgpu_nn_exports::attention_exports::MhaInputPy;
        #[pymodule_export]
        use wgpu_nn_exports::attention_exports::MhaOutputPy;
        #[pymodule_export]
        use wgpu_nn_exports::attention_exports::MultiHeadAttentionConfigPy;
        #[pymodule_export]
        use wgpu_nn_exports::attention_exports::MultiHeadAttentionPy;
        #[pymodule_export]
        use wgpu_nn_exports::attention_exports::MultiHeadAttentionRecordPy;
    }

    #[pymodule]
    pub mod conv {
        use super::*;

        #[pymodule_export]
        use wgpu_nn_exports::conv_exports::Conv1DConfigPy;
        #[pymodule_export]
        use wgpu_nn_exports::conv_exports::Conv1dPy;
        #[pymodule_export]
        use wgpu_nn_exports::conv_exports::Conv1dRecordPy;
        #[pymodule_export]
        use wgpu_nn_exports::conv_exports::Conv2DConfigPy;
        #[pymodule_export]
        use wgpu_nn_exports::conv_exports::Conv2dPy;
        #[pymodule_export]
        use wgpu_nn_exports::conv_exports::Conv2dRecordPy;
        #[pymodule_export]
        use wgpu_nn_exports::conv_exports::Conv3DConfigPy;
        #[pymodule_export]
        use wgpu_nn_exports::conv_exports::Conv3DPy;
        #[pymodule_export]
        use wgpu_nn_exports::conv_exports::ConvTranspose1dConfigPy;
        #[pymodule_export]
        use wgpu_nn_exports::conv_exports::ConvTranspose1dPy;
        #[pymodule_export]
        use wgpu_nn_exports::conv_exports::ConvTranspose1dRecordPy;
        #[pymodule_export]
        use wgpu_nn_exports::conv_exports::ConvTranspose2dConfigPy;
        #[pymodule_export]
        use wgpu_nn_exports::conv_exports::ConvTranspose2dPy;
        #[pymodule_export]
        use wgpu_nn_exports::conv_exports::ConvTranspose2dRecordPy;
        #[pymodule_export]
        use wgpu_nn_exports::conv_exports::ConvTranspose3dConfigPy;
        #[pymodule_export]
        use wgpu_nn_exports::conv_exports::ConvTranspose3dPy;
        #[pymodule_export]
        use wgpu_nn_exports::conv_exports::ConvTranspose3dRecordPy;
        #[pymodule_export]
        use wgpu_nn_exports::conv_exports::DeformConv2dConfigPy;
        #[pymodule_export]
        use wgpu_nn_exports::conv_exports::DeformConv2dPy;
        #[pymodule_export]
        use wgpu_nn_exports::conv_exports::DeformConv2dRecordPy;
    }

    #[pymodule]
    pub mod gru {
        use super::*;

        #[pymodule_export]
        use wgpu_nn_exports::gru_exports::GruConfigPy;
        #[pymodule_export]
        use wgpu_nn_exports::gru_exports::GruPy;
        #[pymodule_export]
        use wgpu_nn_exports::gru_exports::GruRecordPy;
    }

    #[pymodule]
    pub mod interpolate {
        use super::*;

        #[pymodule_export]
        use super::common_nn_exports::interpolate_exports::Interpolate1dConfigPy;
        #[pymodule_export]
        use super::common_nn_exports::interpolate_exports::Interpolate1dPy;
        #[pymodule_export]
        use super::common_nn_exports::interpolate_exports::Interpolate2dConfigPy;
        #[pymodule_export]
        use super::common_nn_exports::interpolate_exports::Interpolate2dPy;
        #[pymodule_export]
        use super::common_nn_exports::interpolate_exports::InterpolateModePy;
    }

    #[pymodule]
    pub mod pool {
        use super::*;
        // use super::common_nn_exports::pool_exports::PyAdaptiveAvgPool1d as AdaptiveAvgPool1d_Py;

        #[pymodule_export]
        use super::common_nn_exports::pool_exports::AdaptiveAvgPool1dPy;

        #[pymodule_export]
        use super::common_nn_exports::pool_exports::AdaptiveAvgPool1dConfigPy;

        #[pymodule_export]
        use super::common_nn_exports::pool_exports::AdaptiveAvgPool2dPy;

        #[pymodule_export]
        use super::common_nn_exports::pool_exports::AdaptiveAvgPool2dConfigPy;

        #[pymodule_export]
        use super::common_nn_exports::pool_exports::AvgPool1dPy;
    }

    #[pymodule]
    pub mod transformer {
        use super::*;
        use burn::nn::transformer::*;

        /// Applies the position-wise feed-forward network to the input tensor from the paper [`Attention Is All You Need`](https://arxiv.org/pdf/1706.03762v7).
        #[pyclass]
        pub struct PositionWiseFeedForwardPy {
            pub inner: PositionWiseFeedForward<Wgpu>,
        }

        #[pymodule_export]
        use wgpu_nn_exports::transformer_exports::PositionWiseFeedForwardConfigPy;
        #[pymodule_export]
        use wgpu_nn_exports::transformer_exports::PositionWiseFeedForwardRecordPy;
        #[pymodule_export]
        use wgpu_nn_exports::transformer_exports::TransformerDecoderAutoregressiveCachePy;
        #[pymodule_export]
        use wgpu_nn_exports::transformer_exports::TransformerDecoderConfigPy;
        #[pymodule_export]
        use wgpu_nn_exports::transformer_exports::TransformerDecoderInputPy;
        #[pymodule_export]
        use wgpu_nn_exports::transformer_exports::TransformerDecoderLayerPy;
        #[pymodule_export]
        use wgpu_nn_exports::transformer_exports::TransformerDecoderLayerRecordPy;
        #[pymodule_export]
        use wgpu_nn_exports::transformer_exports::TransformerDecoderPy;
        #[pymodule_export]
        use wgpu_nn_exports::transformer_exports::TransformerDecoderRecordPy;
        #[pymodule_export]
        use wgpu_nn_exports::transformer_exports::TransformerEncoderAutoregressiveCachePy;
        #[pymodule_export]
        use wgpu_nn_exports::transformer_exports::TransformerEncoderInputPy;
        #[pymodule_export]
        use wgpu_nn_exports::transformer_exports::TransformerEncoderLayerPy;
        #[pymodule_export]
        use wgpu_nn_exports::transformer_exports::TransformerEncoderLayerRecordPy;
        #[pymodule_export]
        use wgpu_nn_exports::transformer_exports::TransformerEncoderPy;
        #[pymodule_export]
        use wgpu_nn_exports::transformer_exports::TransformerEncoderRecordPy;

        implement_send_and_sync!(PositionWiseFeedForwardPy);
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
    use ndarray_nn_exports::EmbeddingPy;
    #[pymodule_export]
    use ndarray_nn_exports::GateControllerPy;
    #[pymodule_export]
    use ndarray_nn_exports::GeLuPy;
    #[pymodule_export]
    use ndarray_nn_exports::GroupNormPy;
    #[pymodule_export]
    use ndarray_nn_exports::HardSigmoidPy;
    #[pymodule_export]
    use ndarray_nn_exports::InitializerPy;
    #[pymodule_export]
    use ndarray_nn_exports::InstanceNormConfigPy;
    #[pymodule_export]
    use ndarray_nn_exports::InstanceNormPy;
    #[pymodule_export]
    use ndarray_nn_exports::InstanceNormRecordPy;
    #[pymodule_export]
    use ndarray_nn_exports::LeakyReluConfigPy;
    #[pymodule_export]
    use ndarray_nn_exports::LeakyReluPy;
    #[pymodule_export]
    use ndarray_nn_exports::LstmConfigPy;
    #[pymodule_export]
    use ndarray_nn_exports::LstmPy;
    #[pymodule_export]
    use ndarray_nn_exports::LstmRecordPy;
    #[pymodule_export]
    use ndarray_nn_exports::PReluPy;
    #[pymodule_export]
    use ndarray_nn_exports::PReluRecordPy;
    #[pymodule_export]
    use ndarray_nn_exports::PaddingConfig1dPy;
    #[pymodule_export]
    use ndarray_nn_exports::PaddingConfig2dPy;
    #[pymodule_export]
    use ndarray_nn_exports::PaddingConfig3dPy;
    #[pymodule_export]
    use ndarray_nn_exports::PositionalEncodingPy;
    #[pymodule_export]
    use ndarray_nn_exports::RmsNormConfigPy;
    #[pymodule_export]
    use ndarray_nn_exports::RmsNormPy;
    #[pymodule_export]
    use ndarray_nn_exports::RmsNormRecordPy;
    #[pymodule_export]
    use ndarray_nn_exports::RotaryEncodingPy;
    #[pymodule_export]
    use ndarray_nn_exports::RotaryEncodingRecordPy;
    #[pymodule_export]
    use ndarray_nn_exports::SigmoidPy;
    #[pymodule_export]
    use ndarray_nn_exports::SwiGluConfigPy;
    #[pymodule_export]
    use ndarray_nn_exports::SwiGluPy;
    #[pymodule_export]
    use ndarray_nn_exports::TanhPy;
    #[pymodule_export]
    use ndarray_nn_exports::Unfold4dConfigPy;
    #[pymodule_export]
    use ndarray_nn_exports::Unfold4dPy;

    /// Applies Linear transformation over a tensor
    #[pyclass]
    #[derive(Debug)]
    #[repr(transparent)]
    pub struct LinearPy {
        pub inner: Linear<NdArray>,
    }

    /// Offers an avenue to configure the BatchNorm layer
    #[pyclass]
    #[derive(Debug)]
    #[repr(transparent)]
    pub struct BatchNormConfigPy(pub BatchNormConfig);

    //[`TODO`] @kwach this `BatchNormRecord` is generic with two arguments; @kwach FIX this
    /// The record type for the BatchNorm module
    #[pyclass]
    #[repr(transparent)]
    pub struct BatchNormRecordPy {
        pub inner: BatchNormRecord<NdArray, 1>,
    }

    /// The implementation of the Bidirectional LSTM module.
    #[pyclass]
    #[repr(transparent)]
    pub struct BiLSTMPy {
        pub inner: BiLstm<NdArray>,
    }

    /// Configuraation to build the BiLSTM module
    #[pyclass]
    pub struct BiLSTMConfigPy(pub BiLstmConfig);

    /// The Dropout layer; set at random elements of the input tensor to zero during training.
    #[pyclass]
    #[derive(Debug)]
    #[repr(transparent)]
    pub struct DropoutPy(pub Dropout);

    implement_send_and_sync!(LinearPy);
    implement_send_and_sync!(BatchNormRecordPy);
    implement_send_and_sync!(BiLSTMPy);

    /// Loss module that exposes various loss functions
    #[pymodule]
    pub mod loss {
        use super::*;

        /// The BinaryCrossEntropyLoss; calculate oss from input logits and targets
        #[pyclass]
        pub struct BinaryCrossEntropyPy {
            pub inner: nn::loss::BinaryCrossEntropyLoss<NdArray>,
        }

        /// Configuration to build the BinaryCrossEntropyLoss
        #[pyclass]
        pub struct BinaryCrossEntropyConfigPy(pub nn::loss::BinaryCrossEntropyLossConfig);

        /// calculate cross entropy loss from input logits to target
        #[pyclass]
        pub struct CrossEntropyLossPy {
            pub inner: nn::loss::CrossEntropyLoss<NdArray>,
        }

        /// Calculate the HuberLoss between inputs and target
        #[pyclass]
        pub struct HuberLossPy(pub nn::loss::HuberLoss);

        /// Configuration to build the HuberLoss
        #[pyclass]
        pub struct HuberLossConfigPy(pub nn::loss::HuberLossConfig);

        /// Calculate the mean squared error loss from the input logits and the targets.
        #[pyclass]
        pub struct MseLoss(pub nn::loss::MseLoss);

        /// Negative Log Likelihood (NLL) loss with a Poisson distribution assumption for the target.
        #[pyclass]
        pub struct PoissonLoss(pub nn::loss::PoissonNllLoss);

        /// Configuration to calculate the PoissonLoss
        #[pyclass]
        pub struct PoissonLossConfig(pub nn::loss::PoissonNllLossConfig);

        implement_send_and_sync!(BinaryCrossEntropyPy);
        implement_send_and_sync!(CrossEntropyLossPy);
    }

    #[pymodule]
    pub mod attention {
        use super::*;

        #[pymodule_export]
        use ndarray_nn_exports::attention_exports::GeneratePaddingMaskPy;
        #[pymodule_export]
        use ndarray_nn_exports::attention_exports::MhaCachePy;
        #[pymodule_export]
        use ndarray_nn_exports::attention_exports::MhaInputPy;
        #[pymodule_export]
        use ndarray_nn_exports::attention_exports::MhaOutputPy;
        #[pymodule_export]
        use ndarray_nn_exports::attention_exports::MultiHeadAttentionConfigPy;
        #[pymodule_export]
        use ndarray_nn_exports::attention_exports::MultiHeadAttentionPy;
        #[pymodule_export]
        use ndarray_nn_exports::attention_exports::MultiHeadAttentionRecordPy;
    }

    #[pymodule]
    pub mod conv {
        use super::*;

        #[pymodule_export]
        use ndarray_nn_exports::conv_exports::Conv1DConfigPy;
        #[pymodule_export]
        use ndarray_nn_exports::conv_exports::Conv1dPy;
        #[pymodule_export]
        use ndarray_nn_exports::conv_exports::Conv1dRecordPy;
        #[pymodule_export]
        use ndarray_nn_exports::conv_exports::Conv2DConfigPy;
        #[pymodule_export]
        use ndarray_nn_exports::conv_exports::Conv2dPy;
        #[pymodule_export]
        use ndarray_nn_exports::conv_exports::Conv2dRecordPy;
        #[pymodule_export]
        use ndarray_nn_exports::conv_exports::Conv3DConfigPy;
        #[pymodule_export]
        use ndarray_nn_exports::conv_exports::Conv3DPy;
        #[pymodule_export]
        use ndarray_nn_exports::conv_exports::ConvTranspose1dConfigPy;
        #[pymodule_export]
        use ndarray_nn_exports::conv_exports::ConvTranspose1dPy;
        #[pymodule_export]
        use ndarray_nn_exports::conv_exports::ConvTranspose1dRecordPy;
        #[pymodule_export]
        use ndarray_nn_exports::conv_exports::ConvTranspose2dConfigPy;
        #[pymodule_export]
        use ndarray_nn_exports::conv_exports::ConvTranspose2dPy;
        #[pymodule_export]
        use ndarray_nn_exports::conv_exports::ConvTranspose2dRecordPy;
        #[pymodule_export]
        use ndarray_nn_exports::conv_exports::ConvTranspose3dConfigPy;
        #[pymodule_export]
        use ndarray_nn_exports::conv_exports::ConvTranspose3dPy;
        #[pymodule_export]
        use ndarray_nn_exports::conv_exports::ConvTranspose3dRecordPy;
        #[pymodule_export]
        use ndarray_nn_exports::conv_exports::DeformConv2dConfigPy;
        #[pymodule_export]
        use ndarray_nn_exports::conv_exports::DeformConv2dPy;
        #[pymodule_export]
        use ndarray_nn_exports::conv_exports::DeformConv2dRecordPy;
    }

    #[pymodule]
    pub mod gru {
        use super::*;

        #[pymodule_export]
        use ndarray_nn_exports::gru_exports::GruConfigPy;
        #[pymodule_export]
        use ndarray_nn_exports::gru_exports::GruPy;
        #[pymodule_export]
        use ndarray_nn_exports::gru_exports::GruRecordPy;
    }

    #[pymodule]
    pub mod interpolate {
        use super::*;

        #[pymodule_export]
        use super::common_nn_exports::interpolate_exports::Interpolate1dConfigPy;
        #[pymodule_export]
        use super::common_nn_exports::interpolate_exports::Interpolate1dPy;
        #[pymodule_export]
        use super::common_nn_exports::interpolate_exports::Interpolate2dConfigPy;
        #[pymodule_export]
        use super::common_nn_exports::interpolate_exports::Interpolate2dPy;
        #[pymodule_export]
        use super::common_nn_exports::interpolate_exports::InterpolateModePy;
    }

    #[pymodule]
    pub mod pool {
        use super::*;

        #[pymodule_export]
        use super::common_nn_exports::pool_exports::AdaptiveAvgPool1dPy;

        #[pymodule_export]
        use super::common_nn_exports::pool_exports::AdaptiveAvgPool1dConfigPy;

        #[pymodule_export]
        use super::common_nn_exports::pool_exports::AdaptiveAvgPool2dPy;

        #[pymodule_export]
        use super::common_nn_exports::pool_exports::AdaptiveAvgPool2dConfigPy;

        #[pymodule_export]
        use super::common_nn_exports::pool_exports::AvgPool1dPy;
    }

    #[pymodule]
    pub mod transformer {
        use super::*;
        use burn::nn::transformer::*;

        /// Applies the position-wise feed-forward network to the input tensor from the paper [`Attention Is All You Need`](https://arxiv.org/pdf/1706.03762v7).
        #[pyclass]
        pub struct PositionWiseFeedForwardPy {
            pub inner: PositionWiseFeedForward<NdArray>,
        }

        #[pymodule_export]
        use ndarray_nn_exports::transformer_exports::PositionWiseFeedForwardConfigPy;
        #[pymodule_export]
        use ndarray_nn_exports::transformer_exports::PositionWiseFeedForwardRecordPy;
        #[pymodule_export]
        use ndarray_nn_exports::transformer_exports::TransformerDecoderAutoregressiveCachePy;
        #[pymodule_export]
        use ndarray_nn_exports::transformer_exports::TransformerDecoderConfigPy;
        #[pymodule_export]
        use ndarray_nn_exports::transformer_exports::TransformerDecoderInputPy;
        #[pymodule_export]
        use ndarray_nn_exports::transformer_exports::TransformerDecoderLayerPy;
        #[pymodule_export]
        use ndarray_nn_exports::transformer_exports::TransformerDecoderLayerRecordPy;
        #[pymodule_export]
        use ndarray_nn_exports::transformer_exports::TransformerDecoderPy;
        #[pymodule_export]
        use ndarray_nn_exports::transformer_exports::TransformerDecoderRecordPy;
        #[pymodule_export]
        use ndarray_nn_exports::transformer_exports::TransformerEncoderAutoregressiveCachePy;
        #[pymodule_export]
        use ndarray_nn_exports::transformer_exports::TransformerEncoderInputPy;
        #[pymodule_export]
        use ndarray_nn_exports::transformer_exports::TransformerEncoderLayerPy;
        #[pymodule_export]
        use ndarray_nn_exports::transformer_exports::TransformerEncoderLayerRecordPy;
        #[pymodule_export]
        use ndarray_nn_exports::transformer_exports::TransformerEncoderPy;
        #[pymodule_export]
        use ndarray_nn_exports::transformer_exports::TransformerEncoderRecordPy;

        implement_send_and_sync!(PositionWiseFeedForwardPy);
    }
}

// [`TODO`] Item types unimmplemented
// [`TODO`] Implement configuration methods as python functions
