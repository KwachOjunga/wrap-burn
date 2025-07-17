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
    use wgpu_nn_exports::PyLstmConfig;
    #[pymodule_export]
    use wgpu_nn_exports::PyLstmRecord;
    #[pymodule_export]
    use wgpu_nn_exports::PyPRelu;
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

        #[pymodule_export]
        use wgpu_nn_exports::attention_exports::PyGeneratePaddingMask;
        #[pymodule_export]
        use wgpu_nn_exports::attention_exports::PyMhaCache;
        #[pymodule_export]
        use wgpu_nn_exports::attention_exports::PyMhaInput;
        #[pymodule_export]
        use wgpu_nn_exports::attention_exports::PyMhaOutput;
        #[pymodule_export]
        use wgpu_nn_exports::attention_exports::PyMultiHeadAttention;
        #[pymodule_export]
        use wgpu_nn_exports::attention_exports::PyMultiHeadAttentionConfig;
        #[pymodule_export]
        use wgpu_nn_exports::attention_exports::PyMultiHeadAttentionRecord;
    }

    #[pymodule]
    pub mod conv {
        use super::*;

        #[pymodule_export]
        use wgpu_nn_exports::conv_exports::PyConv1DConfig;
        #[pymodule_export]
        use wgpu_nn_exports::conv_exports::PyConv1d;
        #[pymodule_export]
        use wgpu_nn_exports::conv_exports::PyConv1dRecord;
        #[pymodule_export]
        use wgpu_nn_exports::conv_exports::PyConv2DConfig;
        #[pymodule_export]
        use wgpu_nn_exports::conv_exports::PyConv2d;
        #[pymodule_export]
        use wgpu_nn_exports::conv_exports::PyConv2dRecord;
        #[pymodule_export]
        use wgpu_nn_exports::conv_exports::PyConv3D;
        #[pymodule_export]
        use wgpu_nn_exports::conv_exports::PyConv3DConfig;
        #[pymodule_export]
        use wgpu_nn_exports::conv_exports::PyConvTranspose1d;
        #[pymodule_export]
        use wgpu_nn_exports::conv_exports::PyConvTranspose1dConfig;
        #[pymodule_export]
        use wgpu_nn_exports::conv_exports::PyConvTranspose1dRecord;
        #[pymodule_export]
        use wgpu_nn_exports::conv_exports::PyConvTranspose2d;
        #[pymodule_export]
        use wgpu_nn_exports::conv_exports::PyConvTranspose2dConfig;
        #[pymodule_export]
        use wgpu_nn_exports::conv_exports::PyConvTranspose2dRecord;
        #[pymodule_export]
        use wgpu_nn_exports::conv_exports::PyConvTranspose3d;
        #[pymodule_export]
        use wgpu_nn_exports::conv_exports::PyConvTranspose3dConfig;
        #[pymodule_export]
        use wgpu_nn_exports::conv_exports::PyConvTranspose3dRecord;
        #[pymodule_export]
        use wgpu_nn_exports::conv_exports::PyDeformConv2d;
        #[pymodule_export]
        use wgpu_nn_exports::conv_exports::PyDeformConv2dConfig;
        #[pymodule_export]
        use wgpu_nn_exports::conv_exports::PyDeformConv2dRecord;
    }

    #[pymodule]
    pub mod gru {
        use super::*;
        
        #[pymodule_export]
        use wgpu_nn_exports::gru_exports::PyGru;
        #[pymodule_export]
        use wgpu_nn_exports::gru_exports::PyGruConfig;
        #[pymodule_export]
        use wgpu_nn_exports::gru_exports::PyGruRecord;
        
    }

    #[pymodule]
    pub mod interpolate {
        use super::*;
        
        #[pymodule_export]
        use super::common_nn_exports::interpolate_exports::PyInterpolate1d;
        #[pymodule_export]
        use super::common_nn_exports::interpolate_exports::PyInterpolate1dConfig;
        #[pymodule_export]
        use super::common_nn_exports::interpolate_exports::PyInterpolate2d;
        #[pymodule_export]
        use super::common_nn_exports::interpolate_exports::PyInterpolate2dConfig;
        #[pymodule_export]
        use super::common_nn_exports::interpolate_exports::PyInterpolateMode;
    }

    

    #[pymodule]
    pub mod pool {
        use super::*;

        #[pymodule_export]
        use super::common_nn_exports::pool_exports::PyAdaptiveAvgPool1d;

        #[pymodule_export]
        use super::common_nn_exports::pool_exports::PyAdaptiveAvgPool1dConfig;

        #[pymodule_export]
        use super::common_nn_exports::pool_exports::PyAdaptiveAvgPool2d;

        #[pymodule_export]
        use super::common_nn_exports::pool_exports::PyAdaptiveAvgPool2dConfig;

        #[pymodule_export]
        use super::common_nn_exports::pool_exports::PyAvgPool1d;
    }

    #[pymodule]
    pub mod transformer {
        use super::*;
        use burn::nn::transformer::*;

        /// Applies the position-wise feed-forward network to the input tensor from the paper [`Attention Is All You Need`](https://arxiv.org/pdf/1706.03762v7).
        #[pyclass]
        pub struct PyPositionWiseFeedForward {
            pub inner: PositionWiseFeedForward<Wgpu>,
        }

        #[pymodule_export]
        use wgpu_nn_exports::transformer_exports::PyPositionWiseFeedForwardConfig;
        #[pymodule_export]
        use wgpu_nn_exports::transformer_exports::PyPositionWiseFeedForwardRecord;
        #[pymodule_export]
        use wgpu_nn_exports::transformer_exports::PyTransformerDecoder;
        #[pymodule_export]
        use wgpu_nn_exports::transformer_exports::PyTransformerDecoderAutoregressiveCache;
        #[pymodule_export]
        use wgpu_nn_exports::transformer_exports::PyTransformerDecoderConfig;
        #[pymodule_export]
        use wgpu_nn_exports::transformer_exports::PyTransformerDecoderInput;
        #[pymodule_export]
        use wgpu_nn_exports::transformer_exports::PyTransformerDecoderLayer;
        #[pymodule_export]
        use wgpu_nn_exports::transformer_exports::PyTransformerDecoderLayerRecord;
        #[pymodule_export]
        use wgpu_nn_exports::transformer_exports::PyTransformerDecoderRecord;
        #[pymodule_export]
        use wgpu_nn_exports::transformer_exports::PyTransformerEncoder;
        #[pymodule_export]
        use wgpu_nn_exports::transformer_exports::PyTransformerEncoderAutoregressiveCache;
        #[pymodule_export]
        use wgpu_nn_exports::transformer_exports::PyTransformerEncoderInput;
        #[pymodule_export]
        use wgpu_nn_exports::transformer_exports::PyTransformerEncoderLayer;
        #[pymodule_export]
        use wgpu_nn_exports::transformer_exports::PyTransformerEncoderLayerRecord;
        #[pymodule_export]
        use wgpu_nn_exports::transformer_exports::PyTransformerEncoderRecord;

        implement_send_and_sync!(PyPositionWiseFeedForward);
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
    use ndarray_nn_exports::PyLstmConfig;
    #[pymodule_export]
    use ndarray_nn_exports::PyLstmRecord;
    #[pymodule_export]
    use ndarray_nn_exports::PyPRelu;
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

  /// Applies Linear transformation over a tensor
  #[pyclass]
  #[derive(Debug)]
  #[repr(transparent)]
  pub struct PyLinear {
      pub inner: Linear<NdArray>,
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
      pub inner: BatchNormRecord<NdArray, 1>,
  }

  /// The implementation of the Bidirectional LSTM module.
  #[pyclass]
  #[repr(transparent)]
  pub struct PyBiLSTM {
      pub inner: BiLstm<NdArray>,
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
          pub inner: nn::loss::BinaryCrossEntropyLoss<NdArray>,
      }

      /// Configuration to build the BinaryCrossEntropyLoss
      #[pyclass]
      pub struct PyBinaryCrossEntropyConfig(pub nn::loss::BinaryCrossEntropyLossConfig);

      /// calculate cross entropy loss from input logits to target
      #[pyclass]
      pub struct PyCrossEntropyLoss {
          pub inner: nn::loss::CrossEntropyLoss<NdArray>,
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

        #[pymodule_export]
        use ndarray_nn_exports::attention_exports::PyGeneratePaddingMask;
        #[pymodule_export]
        use ndarray_nn_exports::attention_exports::PyMhaCache;
        #[pymodule_export]
        use ndarray_nn_exports::attention_exports::PyMhaInput;
        #[pymodule_export]
        use ndarray_nn_exports::attention_exports::PyMhaOutput;
        #[pymodule_export]
        use ndarray_nn_exports::attention_exports::PyMultiHeadAttention;
        #[pymodule_export]
        use ndarray_nn_exports::attention_exports::PyMultiHeadAttentionConfig;
        #[pymodule_export]
        use ndarray_nn_exports::attention_exports::PyMultiHeadAttentionRecord;
    }

    #[pymodule]
    pub mod conv {
        use super::*;

        #[pymodule_export]
        use ndarray_nn_exports::conv_exports::PyConv1DConfig;
        #[pymodule_export]
        use ndarray_nn_exports::conv_exports::PyConv1d;
        #[pymodule_export]
        use ndarray_nn_exports::conv_exports::PyConv1dRecord;
        #[pymodule_export]
        use ndarray_nn_exports::conv_exports::PyConv2DConfig;
        #[pymodule_export]
        use ndarray_nn_exports::conv_exports::PyConv2d;
        #[pymodule_export]
        use ndarray_nn_exports::conv_exports::PyConv2dRecord;
        #[pymodule_export]
        use ndarray_nn_exports::conv_exports::PyConv3D;
        #[pymodule_export]
        use ndarray_nn_exports::conv_exports::PyConv3DConfig;
        #[pymodule_export]
        use ndarray_nn_exports::conv_exports::PyConvTranspose1d;
        #[pymodule_export]
        use ndarray_nn_exports::conv_exports::PyConvTranspose1dConfig;
        #[pymodule_export]
        use ndarray_nn_exports::conv_exports::PyConvTranspose1dRecord;
        #[pymodule_export]
        use ndarray_nn_exports::conv_exports::PyConvTranspose2d;
        #[pymodule_export]
        use ndarray_nn_exports::conv_exports::PyConvTranspose2dConfig;
        #[pymodule_export]
        use ndarray_nn_exports::conv_exports::PyConvTranspose2dRecord;
        #[pymodule_export]
        use ndarray_nn_exports::conv_exports::PyConvTranspose3d;
        #[pymodule_export]
        use ndarray_nn_exports::conv_exports::PyConvTranspose3dConfig;
        #[pymodule_export]
        use ndarray_nn_exports::conv_exports::PyConvTranspose3dRecord;
        #[pymodule_export]
        use ndarray_nn_exports::conv_exports::PyDeformConv2d;
        #[pymodule_export]
        use ndarray_nn_exports::conv_exports::PyDeformConv2dConfig;
        #[pymodule_export]
        use ndarray_nn_exports::conv_exports::PyDeformConv2dRecord;
    }

    #[pymodule]
    pub mod gru {
        use super::*;
        
        #[pymodule_export]
        use ndarray_nn_exports::gru_exports::PyGru;
        #[pymodule_export]
        use ndarray_nn_exports::gru_exports::PyGruConfig;
        #[pymodule_export]
        use ndarray_nn_exports::gru_exports::PyGruRecord;
        
    }

    #[pymodule]
    pub mod interpolate {
        use super::*;
        
        #[pymodule_export]
        use super::common_nn_exports::interpolate_exports::PyInterpolate1d;
        #[pymodule_export]
        use super::common_nn_exports::interpolate_exports::PyInterpolate1dConfig;
        #[pymodule_export]
        use super::common_nn_exports::interpolate_exports::PyInterpolate2d;
        #[pymodule_export]
        use super::common_nn_exports::interpolate_exports::PyInterpolate2dConfig;
        #[pymodule_export]
        use super::common_nn_exports::interpolate_exports::PyInterpolateMode;
    }

    #[pymodule]
    pub mod pool {
        use super::*;

        #[pymodule_export]
        use super::common_nn_exports::pool_exports::PyAdaptiveAvgPool1d;

        #[pymodule_export]
        use super::common_nn_exports::pool_exports::PyAdaptiveAvgPool1dConfig;

        #[pymodule_export]
        use super::common_nn_exports::pool_exports::PyAdaptiveAvgPool2d;

        #[pymodule_export]
        use super::common_nn_exports::pool_exports::PyAdaptiveAvgPool2dConfig;

        #[pymodule_export]
        use super::common_nn_exports::pool_exports::PyAvgPool1d;
    }

    #[pymodule]
    pub mod transformer {
        use super::*;
        use burn::nn::transformer::*;

        /// Applies the position-wise feed-forward network to the input tensor from the paper [`Attention Is All You Need`](https://arxiv.org/pdf/1706.03762v7).
        #[pyclass]
        pub struct PyPositionWiseFeedForward {
            pub inner: PositionWiseFeedForward<NdArray>,
        }

        #[pymodule_export]
        use ndarray_nn_exports::transformer_exports::PyPositionWiseFeedForwardConfig;
        #[pymodule_export]
        use ndarray_nn_exports::transformer_exports::PyPositionWiseFeedForwardRecord;
        #[pymodule_export]
        use ndarray_nn_exports::transformer_exports::PyTransformerDecoder;
        #[pymodule_export]
        use ndarray_nn_exports::transformer_exports::PyTransformerDecoderAutoregressiveCache;
        #[pymodule_export]
        use ndarray_nn_exports::transformer_exports::PyTransformerDecoderConfig;
        #[pymodule_export]
        use ndarray_nn_exports::transformer_exports::PyTransformerDecoderInput;
        #[pymodule_export]
        use ndarray_nn_exports::transformer_exports::PyTransformerDecoderLayer;
        #[pymodule_export]
        use ndarray_nn_exports::transformer_exports::PyTransformerDecoderLayerRecord;
        #[pymodule_export]
        use ndarray_nn_exports::transformer_exports::PyTransformerDecoderRecord;
        #[pymodule_export]
        use ndarray_nn_exports::transformer_exports::PyTransformerEncoder;
        #[pymodule_export]
        use ndarray_nn_exports::transformer_exports::PyTransformerEncoderAutoregressiveCache;
        #[pymodule_export]
        use ndarray_nn_exports::transformer_exports::PyTransformerEncoderInput;
        #[pymodule_export]
        use ndarray_nn_exports::transformer_exports::PyTransformerEncoderLayer;
        #[pymodule_export]
        use ndarray_nn_exports::transformer_exports::PyTransformerEncoderLayerRecord;
        #[pymodule_export]
        use ndarray_nn_exports::transformer_exports::PyTransformerEncoderRecord;

        implement_send_and_sync!(PyPositionWiseFeedForward);
    }
}

// [`TODO`] Item types unimmplemented
// [`TODO`] Implement configuration methods as python functions
