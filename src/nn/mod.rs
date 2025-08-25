#![allow(unused)]
#![recursion_limit = "512"]

//! [`pyburn`] attempts to expose burn's modules and methods in a manner that permits it to work
//! as a python interface. This module exposes the [`burn::nn`] module.

use crate::tensor::tensor_error::TensorError;
use crate::{
    for_normal_struct_enums, implement_ndarray_interface, implement_send_and_sync,
    implement_wgpu_interface,
};
use burn::backend::ndarray::NdArrayDevice;
use burn::backend::wgpu::WgpuDevice;
use burn::nn::Linear;
use burn::nn::*;
use burn::prelude::*;
use pyo3::prelude::*;

mod common_nn_exports;
mod ndarray_nn_exports;
mod wgpu_nn_exports;
// I thought send and Sync were implemented automatically??

pub static WGPUDEVICE: WgpuDevice = WgpuDevice::DefaultDevice;
pub static NDARRAYDEVICE: NdArrayDevice = NdArrayDevice::Cpu;

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
    use common_nn_exports::GeLuPy;
    #[pymodule_export]
    use common_nn_exports::HardSigmoidPy;
    #[pymodule_export]
    use common_nn_exports::LeakyReluPy;
    #[pymodule_export]
    use common_nn_exports::PaddingConfig1dPy;
    #[pymodule_export]
    use common_nn_exports::PaddingConfig2dPy;
    #[pymodule_export]
    use common_nn_exports::PaddingConfig3dPy;
    #[pymodule_export]
    use common_nn_exports::SigmoidPy;
    #[pymodule_export]
    use common_nn_exports::TanhPy;
    #[pymodule_export]
    use wgpu_nn_exports::EmbeddingPy;
    #[pymodule_export]
    use wgpu_nn_exports::GateControllerPy;
    #[pymodule_export]
    use wgpu_nn_exports::GroupNormPy;
    #[pymodule_export]
    use wgpu_nn_exports::InstanceNormPy;
    #[pymodule_export]
    use wgpu_nn_exports::InstanceNormRecordPy;
    #[pymodule_export]
    use wgpu_nn_exports::LstmPy;
    #[pymodule_export]
    use wgpu_nn_exports::LstmRecordPy;
    #[pymodule_export]
    use wgpu_nn_exports::PReluPy;
    #[pymodule_export]
    use wgpu_nn_exports::PReluRecordPy;
    #[pymodule_export]
    use wgpu_nn_exports::PositionalEncodingPy;
    #[pymodule_export]
    use wgpu_nn_exports::RmsNormPy;
    #[pymodule_export]
    use wgpu_nn_exports::RmsNormRecordPy;
    #[pymodule_export]
    use wgpu_nn_exports::RotaryEncodingPy;
    #[pymodule_export]
    use wgpu_nn_exports::RotaryEncodingRecordPy;
    #[pymodule_export]
    use wgpu_nn_exports::SwiGluPy;

    // [TODO:] Note the current implementation of this
    #[pymodule_export]
    use crate::tensor::wgpu_base::TensorPy;
    #[pymodule_export]
    use common_nn_exports::Initializer;
    #[pymodule_export]
    use common_nn_exports::Unfold4dPy;

    /// Applies Linear transformation over a tensor
    #[pyclass]
    #[derive(Debug)]
    #[repr(transparent)]
    pub struct LinearPy {
        pub inner: Linear<Wgpu>,
    }

    impl From<Linear<Wgpu>> for LinearPy {
        fn from(inner: Linear<Wgpu>) -> Self {
            LinearPy { inner }
        }
    }

    #[pymethods]
    impl LinearPy {
        #[new]
        #[pyo3(signature = (d_input, d_output, with_bias=None, with_initializer=None))]
        fn new(
            d_input: usize,
            d_output: usize,
            with_bias: Option<bool>,
            with_initializer: Option<Initializer>,
        ) -> Self {
            let bias = match with_bias {
                Some(b) => b,
                None => true,
            };
            let init = match with_initializer {
                Some(init) => match init {
                    Initializer::Constant { value } => {
                        Some(burn::nn::Initializer::Constant { value })
                    }
                    Initializer::One() => Some(burn::nn::Initializer::Ones),
                    Initializer::Zero() => Some(burn::nn::Initializer::Zeros),
                    Initializer::Uniform { min, max } => {
                        Some(burn::nn::Initializer::Uniform { min, max })
                    }
                    Initializer::Normal { mean, std } => {
                        Some(burn::nn::Initializer::Normal { mean, std })
                    }
                    Initializer::KaimingNormal { gain, fan_out_only } => {
                        Some(burn::nn::Initializer::KaimingNormal { gain, fan_out_only })
                    }
                    Initializer::KaimingUniform { gain, fan_out_only } => {
                        Some(burn::nn::Initializer::KaimingUniform { gain, fan_out_only })
                    }
                    Initializer::XavierNormal { gain } => {
                        Some(burn::nn::Initializer::XavierNormal { gain })
                    }
                    Initializer::XavierUniform { gain } => {
                        Some(burn::nn::Initializer::XavierUniform { gain })
                    }
                    Initializer::Orthogonal { gain } => {
                        Some(burn::nn::Initializer::Orthogonal { gain })
                    }
                },
                None => None, /*KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0), fan_out_only:false}*/
            };
            match init {
                Some(init) => LinearConfig::new(d_input, d_output)
                    .with_bias(bias)
                    .with_initializer(init)
                    .init(&WGPUDEVICE)
                    .into(),
                None => LinearConfig::new(d_input, d_output)
                    .with_bias(bias)
                    .init(&WGPUDEVICE)
                    .into(),
            }
        }
        /// forward pass for the Linear layer
        fn forward(
            &self,
            input: crate::tensor::wgpu_base::TensorPy,
        ) -> PyResult<crate::tensor::wgpu_base::TensorPy> {
            match input {
                TensorPy::TensorOne(tensor) => Ok(self.inner.forward(tensor.inner).into()),
                TensorPy::TensorTwo(tensor) => Ok(self.inner.forward(tensor.inner).into()),
                TensorPy::TensorThree(tensor) => Ok(self.inner.forward(tensor.inner).into()),
                TensorPy::TensorFour(tensor) => Ok(self.inner.forward(tensor.inner).into()),
                TensorPy::TensorFive(tensor) => Ok(self.inner.forward(tensor.inner).into()),
                _ => Err(TensorError::NonApplicableMethod.into()),
            }
        }
    }

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
    pub struct BiLstmPy {
        pub inner: BiLstm<Wgpu>,
    }

    impl From<BiLstm<Wgpu>> for BiLstmPy {
        fn from(other: BiLstm<Wgpu>) -> Self {
            Self { inner: other }
        }
    }

    #[pymethods]
    impl BiLstmPy {
        #[new]
        fn new(
            d_input: usize,
            d_hidden: usize,
            bias: bool,
            initializer: Option<crate::nn::common_nn_exports::Initializer>,
        ) -> Self {
            let init = match initializer {
                Some(init) => match init {
                    crate::nn::common_nn_exports::Initializer::Constant { value } => {
                        Some(burn::nn::Initializer::Constant { value })
                    }
                    crate::nn::common_nn_exports::Initializer::One() => {
                        Some(burn::nn::Initializer::Ones)
                    }
                    crate::nn::common_nn_exports::Initializer::Zero() => {
                        Some(burn::nn::Initializer::Zeros)
                    }
                    crate::nn::common_nn_exports::Initializer::Uniform { min, max } => {
                        Some(burn::nn::Initializer::Uniform { min, max })
                    }
                    crate::nn::common_nn_exports::Initializer::Normal { mean, std } => {
                        Some(burn::nn::Initializer::Normal { mean, std })
                    }
                    crate::nn::common_nn_exports::Initializer::KaimingNormal {
                        gain,
                        fan_out_only,
                    } => Some(burn::nn::Initializer::KaimingNormal { gain, fan_out_only }),
                    crate::nn::common_nn_exports::Initializer::KaimingUniform {
                        gain,
                        fan_out_only,
                    } => Some(burn::nn::Initializer::KaimingUniform { gain, fan_out_only }),
                    crate::nn::common_nn_exports::Initializer::XavierNormal { gain } => {
                        Some(burn::nn::Initializer::XavierNormal { gain })
                    }
                    crate::nn::common_nn_exports::Initializer::XavierUniform { gain } => {
                        Some(burn::nn::Initializer::XavierUniform { gain })
                    }
                    crate::nn::common_nn_exports::Initializer::Orthogonal { gain } => {
                        Some(burn::nn::Initializer::Orthogonal { gain })
                    }
                },
                None => None, /*KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0), fan_out_only:false}*/
            };

            match init {
                Some(init) => BiLstmConfig::new(d_input, d_hidden, bias)
                    .with_initializer(init)
                    .init(&WGPUDEVICE)
                    .into(),
                None => BiLstmConfig::new(d_input, d_hidden, bias)
                    .init(&WGPUDEVICE)
                    .into(),
            }
        }

        // [TODO:] @kwach One of the input parametres for this method is LstmStatePy which is not yet implemented and requires a specific
        //               dimension size.
        //                  Figure that out before ou implement this forward method.
        fn forward(&self) {}
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
    implement_send_and_sync!(BiLstmPy);

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
        use wgpu_nn_exports::conv_exports::Conv1dPy;
        #[pymodule_export]
        use wgpu_nn_exports::conv_exports::Conv1dRecordPy;

        #[pymodule_export]
        use wgpu_nn_exports::conv_exports::Conv2dPy;
        #[pymodule_export]
        use wgpu_nn_exports::conv_exports::Conv2dRecordPy;

        #[pymodule_export]
        use wgpu_nn_exports::conv_exports::Conv3DPy;
        #[pymodule_export]
        use wgpu_nn_exports::conv_exports::ConvTranspose1dPy;
        #[pymodule_export]
        use wgpu_nn_exports::conv_exports::ConvTranspose1dRecordPy;
        #[pymodule_export]
        use wgpu_nn_exports::conv_exports::ConvTranspose2dPy;
        #[pymodule_export]
        use wgpu_nn_exports::conv_exports::ConvTranspose2dRecordPy;
        #[pymodule_export]
        use wgpu_nn_exports::conv_exports::ConvTranspose3dPy;
        #[pymodule_export]
        use wgpu_nn_exports::conv_exports::ConvTranspose3dRecordPy;
        #[pymodule_export]
        use wgpu_nn_exports::conv_exports::DeformConv2dPy;
        #[pymodule_export]
        use wgpu_nn_exports::conv_exports::DeformConv2dRecordPy;
    }

    #[pymodule]
    pub mod gru {
        use super::*;

        #[pymodule_export]
        use wgpu_nn_exports::gru_exports::GruPy;
        #[pymodule_export]
        use wgpu_nn_exports::gru_exports::GruRecordPy;
    }

    #[pymodule]
    pub mod interpolate {
        use super::*;

        #[pymodule_export]
        use super::common_nn_exports::interpolate_exports::Interpolate1dPy;
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

        #[pymodule_export]
        use wgpu_nn_exports::transformer_exports::PositionWiseFeedForwardPy;
        // #[pymodule_export]
        // use wgpu_nn_exports::transformer_exports::PositionWiseFeedForwardConfigPy;
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
pub mod ndarray_nn {

    use super::*;
    use burn::backend::ndarray::*;

    #[pymodule_export]
    use crate::tensor::ndarray_base::TensorPy;
    #[pymodule_export]
    use common_nn_exports::GeLuPy;
    #[pymodule_export]
    use common_nn_exports::HardSigmoidPy;
    #[pymodule_export]
    use common_nn_exports::Initializer;
    #[pymodule_export]
    use common_nn_exports::LeakyReluPy;
    #[pymodule_export]
    use common_nn_exports::PaddingConfig1dPy;
    #[pymodule_export]
    use common_nn_exports::PaddingConfig2dPy;
    #[pymodule_export]
    use common_nn_exports::PaddingConfig3dPy;
    #[pymodule_export]
    use common_nn_exports::SigmoidPy;
    #[pymodule_export]
    use common_nn_exports::TanhPy;
    #[pymodule_export]
    use common_nn_exports::Unfold4dPy;
    #[pymodule_export]
    use ndarray_nn_exports::EmbeddingPy;
    #[pymodule_export]
    use ndarray_nn_exports::GateControllerPy;
    #[pymodule_export]
    use ndarray_nn_exports::GroupNormPy;
    #[pymodule_export]
    use ndarray_nn_exports::InstanceNormPy;
    #[pymodule_export]
    use ndarray_nn_exports::InstanceNormRecordPy;
    #[pymodule_export]
    use ndarray_nn_exports::LstmPy;
    #[pymodule_export]
    use ndarray_nn_exports::LstmRecordPy;
    #[pymodule_export]
    use ndarray_nn_exports::PReluPy;
    #[pymodule_export]
    use ndarray_nn_exports::PReluRecordPy;
    #[pymodule_export]
    use ndarray_nn_exports::PositionalEncodingPy;
    #[pymodule_export]
    use ndarray_nn_exports::RmsNormPy;
    #[pymodule_export]
    use ndarray_nn_exports::RmsNormRecordPy;
    #[pymodule_export]
    use ndarray_nn_exports::RotaryEncodingPy;
    #[pymodule_export]
    use ndarray_nn_exports::RotaryEncodingRecordPy;
    #[pymodule_export]
    use ndarray_nn_exports::SwiGluPy;

    /// Applies Linear transformation over a tensor
    #[pyclass]
    #[derive(Debug)]
    #[repr(transparent)]
    pub struct LinearPy {
        pub inner: Linear<NdArray>,
    }

    impl From<Linear<NdArray>> for LinearPy {
        fn from(inner: Linear<NdArray>) -> Self {
            LinearPy { inner }
        }
    }

    #[pymethods]
    impl LinearPy {
        #[new]
        #[pyo3(signature = (d_input, d_output, with_bias=None, with_initializer=None))]
        fn new(
            d_input: usize,
            d_output: usize,
            with_bias: Option<bool>,
            with_initializer: Option<Initializer>,
        ) -> Self {
            let bias = match with_bias {
                Some(b) => b,
                None => true,
            };
            let init = match with_initializer {
                Some(init) => match init {
                    Initializer::Constant { value } => {
                        Some(burn::nn::Initializer::Constant { value })
                    }
                    Initializer::One() => Some(burn::nn::Initializer::Ones),
                    Initializer::Zero() => Some(burn::nn::Initializer::Zeros),
                    Initializer::Uniform { min, max } => {
                        Some(burn::nn::Initializer::Uniform { min, max })
                    }
                    Initializer::Normal { mean, std } => {
                        Some(burn::nn::Initializer::Normal { mean, std })
                    }
                    Initializer::KaimingNormal { gain, fan_out_only } => {
                        Some(burn::nn::Initializer::KaimingNormal { gain, fan_out_only })
                    }
                    Initializer::KaimingUniform { gain, fan_out_only } => {
                        Some(burn::nn::Initializer::KaimingUniform { gain, fan_out_only })
                    }
                    Initializer::XavierNormal { gain } => {
                        Some(burn::nn::Initializer::XavierNormal { gain })
                    }
                    Initializer::XavierUniform { gain } => {
                        Some(burn::nn::Initializer::XavierUniform { gain })
                    }
                    Initializer::Orthogonal { gain } => {
                        Some(burn::nn::Initializer::Orthogonal { gain })
                    }
                },
                None => None, /*KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0), fan_out_only:false}*/
            };
            match init {
                Some(init) => LinearConfig::new(d_input, d_output)
                    .with_bias(bias)
                    .with_initializer(init)
                    .init(&NDARRAYDEVICE)
                    .into(),
                None => LinearConfig::new(d_input, d_output)
                    .with_bias(bias)
                    .init(&NDARRAYDEVICE)
                    .into(),
            }
        }
        /// forward pass for the Linear layer
        fn forward(&self, input: TensorPy) -> PyResult<TensorPy> {
            match input {
                TensorPy::TensorOne(tensor) => Ok(self.inner.forward(tensor.inner).into()),
                TensorPy::TensorTwo(tensor) => Ok(self.inner.forward(tensor.inner).into()),
                TensorPy::TensorThree(tensor) => Ok(self.inner.forward(tensor.inner).into()),
                TensorPy::TensorFour(tensor) => Ok(self.inner.forward(tensor.inner).into()),
                TensorPy::TensorFive(tensor) => Ok(self.inner.forward(tensor.inner).into()),
                _ => Err(TensorError::NonApplicableMethod.into()),
            }
        }
    }

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
    pub struct BiLstmPy {
        pub inner: BiLstm<NdArray>,
    }

    impl From<BiLstm<NdArray>> for BiLstmPy {
        fn from(other: BiLstm<NdArray>) -> Self {
            Self { inner: other }
        }
    }

    #[pymethods]
    impl BiLstmPy {
        #[new]
        fn new(
            d_input: usize,
            d_hidden: usize,
            bias: bool,
            initializer: Option<crate::nn::common_nn_exports::Initializer>,
        ) -> Self {
            let init = match initializer {
                Some(init) => match init {
                    crate::nn::common_nn_exports::Initializer::Constant { value } => {
                        Some(burn::nn::Initializer::Constant { value })
                    }
                    crate::nn::common_nn_exports::Initializer::One() => {
                        Some(burn::nn::Initializer::Ones)
                    }
                    crate::nn::common_nn_exports::Initializer::Zero() => {
                        Some(burn::nn::Initializer::Zeros)
                    }
                    crate::nn::common_nn_exports::Initializer::Uniform { min, max } => {
                        Some(burn::nn::Initializer::Uniform { min, max })
                    }
                    crate::nn::common_nn_exports::Initializer::Normal { mean, std } => {
                        Some(burn::nn::Initializer::Normal { mean, std })
                    }
                    crate::nn::common_nn_exports::Initializer::KaimingNormal {
                        gain,
                        fan_out_only,
                    } => Some(burn::nn::Initializer::KaimingNormal { gain, fan_out_only }),
                    crate::nn::common_nn_exports::Initializer::KaimingUniform {
                        gain,
                        fan_out_only,
                    } => Some(burn::nn::Initializer::KaimingUniform { gain, fan_out_only }),
                    crate::nn::common_nn_exports::Initializer::XavierNormal { gain } => {
                        Some(burn::nn::Initializer::XavierNormal { gain })
                    }
                    crate::nn::common_nn_exports::Initializer::XavierUniform { gain } => {
                        Some(burn::nn::Initializer::XavierUniform { gain })
                    }
                    crate::nn::common_nn_exports::Initializer::Orthogonal { gain } => {
                        Some(burn::nn::Initializer::Orthogonal { gain })
                    }
                },
                None => None, /*KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0), fan_out_only:false}*/
            };

            match init {
                Some(init) => BiLstmConfig::new(d_input, d_hidden, bias)
                    .with_initializer(init)
                    .init(&NDARRAYDEVICE)
                    .into(),
                None => BiLstmConfig::new(d_input, d_hidden, bias)
                    .init(&NDARRAYDEVICE)
                    .into(),
            }
        }

        // [TODO:] @kwach One of the input parametres for this method is LstmStatePy which is not yet implemented and requires a specific
        //               dimension size.
        //                  Figure that out before ou implement this forward method.
        fn forward(&self) {}
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
    implement_send_and_sync!(BiLstmPy);

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
        use ndarray_nn_exports::conv_exports::Conv1dPy;
        #[pymodule_export]
        use ndarray_nn_exports::conv_exports::Conv1dRecordPy;

        #[pymodule_export]
        use ndarray_nn_exports::conv_exports::Conv2dPy;
        #[pymodule_export]
        use ndarray_nn_exports::conv_exports::Conv2dRecordPy;

        #[pymodule_export]
        use ndarray_nn_exports::conv_exports::Conv3DPy;
        #[pymodule_export]
        use ndarray_nn_exports::conv_exports::ConvTranspose1dPy;
        #[pymodule_export]
        use ndarray_nn_exports::conv_exports::ConvTranspose1dRecordPy;
        #[pymodule_export]
        use ndarray_nn_exports::conv_exports::ConvTranspose2dPy;
        #[pymodule_export]
        use ndarray_nn_exports::conv_exports::ConvTranspose2dRecordPy;
        #[pymodule_export]
        use ndarray_nn_exports::conv_exports::ConvTranspose3dPy;
        #[pymodule_export]
        use ndarray_nn_exports::conv_exports::ConvTranspose3dRecordPy;
        #[pymodule_export]
        use ndarray_nn_exports::conv_exports::DeformConv2dPy;
        #[pymodule_export]
        use ndarray_nn_exports::conv_exports::DeformConv2dRecordPy;
    }

    #[pymodule]
    pub mod gru {
        use super::*;

        #[pymodule_export]
        use ndarray_nn_exports::gru_exports::GruPy;
        #[pymodule_export]
        use ndarray_nn_exports::gru_exports::GruRecordPy;
    }

    #[pymodule]
    pub mod interpolate {
        use super::*;

        #[pymodule_export]
        use super::common_nn_exports::interpolate_exports::Interpolate1dPy;
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

        #[pymodule_export]
        use ndarray_nn_exports::transformer_exports::PositionWiseFeedForwardPy;
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
    }
}

// [`TODO`] Item types unimmplemented
// [`TODO`] Implement configuration methods as python functions
