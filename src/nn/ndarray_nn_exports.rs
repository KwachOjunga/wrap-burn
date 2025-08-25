use std::sync::{Arc, Mutex};

use crate::nn::NDARRAYDEVICE;
use crate::nn::common_nn_exports::*;
use crate::tensor::{ndarray_base::TensorPy, tensor_error::TensorError};
use crate::{for_normal_struct_enums, implement_ndarray_interface, implement_send_and_sync};
use burn::nn::Linear;
use burn::nn::{
    BatchNorm, BatchNormConfig, Embedding, EmbeddingConfig, GateController, GroupNorm,
    GroupNormConfig, InstanceNorm, InstanceNormConfig, InstanceNormRecord, LayerNorm,
    LayerNormConfig, LayerNormRecord, Lstm, LstmConfig, LstmRecord, PRelu, PReluConfig,
    PReluRecord, PositionalEncoding, PositionalEncodingConfig, PositionalEncodingRecord, RmsNorm,
    RmsNormConfig, RmsNormRecord, RotaryEncoding, RotaryEncodingConfig, RotaryEncodingRecord,
    SwiGlu, SwiGluConfig, SwiGluRecord, conv::*,
};
use burn::prelude::*;
use pyo3::prelude::*;

// [`TODO`] Update the documentation to reference the papers. Some of us learn through these frameworks.
implement_ndarray_interface!(
    GateControllerPy,
    GateController,
    "A GateController represents a gate in an LSTM cell.\n An LSTM cell generally contains three gates: an input gate, forget gate,\n and output gate. Additionally, cell gate is just used to compute the cell state
    \n An Lstm gate is modeled as two linear transformations. The results of these transformations are used to calculate the gate's output.
     To delve deeper into the whole system of gates and the problems it attempts to solve; i highly recommend [Learning to forget:Continual prediction with LSTM](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=e10f98b86797ebf6c8caea6f54cacbc5a50e8b34)"
);

// This is now neccesary; there is a clash between the // common_nn_exports::Initializer and Initializer in the burn::nn module.
// [TODO] @kwach refactor the new method in GateControllerPy to use the Initializer instead of InitializerPy.
#[pymethods]
impl GateControllerPy {
    // #[staticmethod]
    // pub fn new(input: usize, output: usize, bias: bool, initializer: InitializerPy) -> Self {
    //     GateController::new(input, output, bias, initializer.0, &NDARRAYDEVICE).into()
    // }

    /// yield the gate product given two Tensors of 2 dimensions
    ///
    /// def gate_product(input: TensorPy, output: TensorPy) -> TensorPy :
    ///
    #[pyo3(text_signature = "(input: TensorPy, output: TensorPy -> PyResult<TensorPy>)")]
    pub fn gate_product(&self, input: TensorPy, hidden: TensorPy) -> PyResult<TensorPy> {
        let i = match input {
            TensorPy::TensorTwo(val) => Ok(val),
            _ => Err(TensorError::WrongDimensions),
        }?;
        let o = match hidden {
            TensorPy::TensorTwo(val) => Ok(val),
            _ => Err(TensorError::WrongDimensions),
        }?;
        Ok(self.inner.gate_product(i.inner, o.inner).into())
    }
}

impl From<GateController<NdArray>> for GateControllerPy {
    fn from(other: GateController<NdArray>) -> Self {
        Self { inner: other }
    }
}

implement_ndarray_interface!(
    EmbeddingPy,
    Embedding,
    "Lookup table to store a fix number of vectors."
);

impl From<Embedding<NdArray>> for EmbeddingPy {
    fn from(other: Embedding<NdArray>) -> Self {
        Self { inner: other }
    }
}

#[pymethods]
impl EmbeddingPy {
    #[new]
    #[pyo3(signature = (n_embedding, d_model, initializer = None))]
    fn new(
        n_embedding: usize,
        d_model: usize,
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
                crate::nn::common_nn_exports::Initializer::KaimingNormal { gain, fan_out_only } => {
                    Some(burn::nn::Initializer::KaimingNormal { gain, fan_out_only })
                }
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
            Some(init) => EmbeddingConfig::new(n_embedding, d_model)
                .with_initializer(init)
                .init(&NDARRAYDEVICE)
                .into(),
            None => EmbeddingConfig::new(n_embedding, d_model)
                .init(&NDARRAYDEVICE)
                .into(),
        }
    }

    // [TODO:] @kwach add a Tensor struct to accomodate the Int type for this forward method

    // fn forward(&self, input: TensorPy) -> PyResult<TensorPy> {
    //     match input {
    //         TensorPy::TensorTwo(tensor) => {
    //             Ok(self.inner.forward(tensor.inner).into())
    //         },
    //         _ => Err(TensorError::NonApplicableMethod.into()),
    //     }
    // }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct BatchNormPy {
    inner: BatchNorm<NdArray, 0>,
}

implement_send_and_sync!(BatchNormPy);

impl From<BatchNorm<NdArray, 0>> for BatchNormPy {
    fn from(other: BatchNorm<NdArray, 0>) -> Self {
        Self { inner: other }
    }
}

// [TODO:] Complete the BatchNormPy class to include the necessary methods and attributes.

#[pymethods]
impl BatchNormPy {
    #[new]
    #[pyo3(signature = (num_features, epsilon = Some(1e-5), momentum = Some(0.1)))]
    fn new(num_features: usize, epsilon: Option<f64>, momentum: Option<f64>) -> Self {
        let epsilon = epsilon.unwrap_or(1e-5);
        let momentum = momentum.unwrap_or(0.1);
        let batch_norm: BatchNorm<NdArray, 0> = BatchNormConfig::new(num_features)
            .with_epsilon(epsilon)
            .with_momentum(momentum)
            .init(&NDARRAYDEVICE);

        batch_norm.into()
    }

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

    fn num_params(&self) -> usize {
        self.inner.num_params()
    }
}

implement_ndarray_interface!(
    GroupNormPy,
    GroupNorm,
    "Applies Group Normalization over a mini-batch of inputs"
);

impl From<GroupNorm<NdArray>> for GroupNormPy {
    fn from(other: GroupNorm<NdArray>) -> Self {
        Self { inner: other }
    }
}

// [TODO:]  @kwach implement a method to save the configuration to a file

#[pymethods]
impl GroupNormPy {
    #[new]
    #[pyo3(signature = (num_groups, num_channels, epsilon = Some(1e-5), affine = Some(true)))]
    fn new(
        num_groups: usize,
        num_channels: usize,
        epsilon: Option<f64>,
        affine: Option<bool>,
    ) -> Self {
        let epsilon = epsilon.unwrap_or(1e-5);
        let affine = affine.unwrap_or(true);
        GroupNormConfig::new(num_groups, num_channels)
            .with_epsilon(epsilon)
            .with_affine(affine)
            .init(&NDARRAYDEVICE)
            .into()
    }

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
implement_ndarray_interface!(
    InstanceNormPy,
    InstanceNorm,
    "Applies Instance Normalization over a tensor"
);

impl From<InstanceNorm<NdArray>> for InstanceNormPy {
    fn from(other: InstanceNorm<NdArray>) -> Self {
        Self { inner: other }
    }
}

#[pymethods]
impl InstanceNormPy {
    #[new]
    #[pyo3(signature = (num_channels, epsilon = Some(1e-5), affine = Some(true)))]
    fn new(num_channels: usize, epsilon: Option<f64>, affine: Option<bool>) -> Self {
        let epsilon = epsilon.unwrap_or(1e-5);
        let affine = affine.unwrap_or(true);
        InstanceNormConfig::new(num_channels)
            .with_epsilon(epsilon)
            .with_affine(affine)
            .init(&NDARRAYDEVICE)
            .into()
    }

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

implement_ndarray_interface!(
    InstanceNormRecordPy,
    InstanceNormRecord,
    "Record type of the InstanceNorm module"
);
implement_ndarray_interface!(
    LayerNormPy,
    LayerNorm,
    "Applies Layer Normalization over a tensor"
);

impl From<LayerNorm<NdArray>> for LayerNormPy {
    fn from(other: LayerNorm<NdArray>) -> Self {
        Self { inner: other }
    }
}

#[pymethods]
impl LayerNormPy {
    #[new]
    #[pyo3(signature = (d_model, epsilon = None))]
    fn new(d_model: usize, epsilon: Option<f64>) -> Self {
        let epsilon = epsilon.unwrap_or(1e-5);
        LayerNormConfig::new(d_model)
            .with_epsilon(epsilon)
            .init(&NDARRAYDEVICE)
            .into()
    }

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

implement_ndarray_interface!(
    LayerNormRecordPy,
    LayerNormRecord,
    "Record type of the LayerNorm record"
);
// implement_ndarray_interface!(PyLinearRecord, LinearRecord);
implement_ndarray_interface!(
    LstmPy,
    Lstm,
    "The Lstm module. This implementation is for a unidirectional, stateless, Lstm"
);

impl From<Lstm<NdArray>> for LstmPy {
    fn from(other: Lstm<NdArray>) -> Self {
        Self { inner: other }
    }
}

// [TODO:] @kwach Implement the LstmState class to allow its use in the Lstm layer.

#[pymethods]
impl LstmPy {
    #[new]
    #[pyo3(signature = (d_input, d_hidden, bias, initializer = None))]
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
                crate::nn::common_nn_exports::Initializer::KaimingNormal { gain, fan_out_only } => {
                    Some(burn::nn::Initializer::KaimingNormal { gain, fan_out_only })
                }
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
            Some(init) => LstmConfig::new(d_input, d_hidden, bias)
                .with_initializer(init)
                .init(&NDARRAYDEVICE)
                .into(),
            None => LstmConfig::new(d_input, d_hidden, bias)
                .init(&NDARRAYDEVICE)
                .into(),
        }
    }
}

implement_ndarray_interface!(LstmRecordPy, LstmRecord, "Record type of the Lstm module");
implement_ndarray_interface!(PReluPy, PRelu, "Parametric Relu Layer");

impl From<PRelu<NdArray>> for PReluPy {
    fn from(other: PRelu<NdArray>) -> Self {
        Self { inner: other }
    }
}

#[pymethods]
impl PReluPy {
    #[new]
    #[pyo3(signature = (num_parameters = None, alpha = None))]
    fn new(num_parameters: Option<usize>, alpha: Option<f64>) -> Self {
        let param = match num_parameters {
            Some(n) => Some(PReluConfig::new().with_num_parameters(n)),
            None => None,
        };
        let alpha = alpha.unwrap_or(0.25);
        match param {
            Some(param) => param.with_alpha(alpha).init(&NDARRAYDEVICE).into(),
            None => PReluConfig::new()
                .with_alpha(alpha)
                .init(&NDARRAYDEVICE)
                .into(),
        }
    }

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
implement_ndarray_interface!(
    PReluRecordPy,
    PReluRecord,
    "record type of the PRelu module"
);
implement_ndarray_interface!(PositionalEncodingPy, PositionalEncoding, "
Positional encoding layer for transformer models \n This layer adds positional information to the input embeddings,\nallowing the transformer model to take into account the order of the sequence.\n The positional encoding is added to the input embeddings by computing\n a set of sinusoidal functions with different frequencies and phases.");

impl From<PositionalEncoding<NdArray>> for PositionalEncodingPy {
    fn from(other: PositionalEncoding<NdArray>) -> Self {
        Self { inner: other }
    }
}

#[pymethods]
impl PositionalEncodingPy {
    #[new]
    #[pyo3(signature = (d_model, max_sequence_size = 5000, max_timescale = 10_000))]
    fn new(d_model: usize, max_sequence_size: Option<usize>, max_timescale: Option<usize>) -> Self {
        let max_sequence_size = max_sequence_size.unwrap_or(5000);
        let max_timescale = max_timescale.unwrap_or(10_000);
        PositionalEncodingConfig::new(d_model)
            .with_max_sequence_size(max_sequence_size)
            .with_max_timescale(max_timescale)
            .init(&NDARRAYDEVICE)
            .into()
    }

    fn forward(&self, input: TensorPy) -> PyResult<TensorPy> {
        match input {
            TensorPy::TensorThree(tensor) => Ok(self.inner.forward(tensor.inner).into()),
            _ => Err(TensorError::NonApplicableMethod.into()),
        }
    }
}
implement_ndarray_interface!(
    PositionalEncodingRecordPy,
    PositionalEncodingRecord,
    "Record type of the PositionalEncoding module"
);
implement_ndarray_interface!(
    RmsNormPy,
    RmsNorm,
    "Applies RMS Normalization over an input tensor along the last dimension"
);

impl From<RmsNorm<NdArray>> for RmsNormPy {
    fn from(other: RmsNorm<NdArray>) -> Self {
        Self { inner: other }
    }
}

#[pymethods]
impl RmsNormPy {
    #[new]
    #[pyo3(signature = (d_model, eps = 1e-5))]
    fn new(d_model: usize, eps: Option<f64>) -> Self {
        let eps = eps.unwrap_or(1e-5);
        RmsNormConfig::new(d_model)
            .with_epsilon(eps)
            .init(&NDARRAYDEVICE)
            .into()
    }

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

implement_ndarray_interface!(
    RmsNormRecordPy,
    RmsNormRecord,
    "Record type of the RmsNormRecord"
);
implement_ndarray_interface!(
    RotaryEncodingPy,
    RotaryEncoding,
    "A module that applies rotary positional encoding to a tensor.\n Rotary Position Encoding or Embedding (RoPE), is a type of \nposition embedding which encodes absolute positional\n information with rotation matrix and naturally incorporates explicit relative \nposition dependency in self-attention formulation."
);
impl From<RotaryEncoding<NdArray>> for RotaryEncodingPy {
    fn from(other: RotaryEncoding<NdArray>) -> Self {
        Self { inner: other }
    }
}

// [TOOD:] @kwach There is a method to implement a rotary encoding layer
// that takes a function whose input is a tensor of dim 1 and returns a temsor of similar dimensions.

#[pymethods]
impl RotaryEncodingPy {
    #[new]
    #[pyo3(signature = (max_sequence_length, d_model, theta = 10000.0))]
    fn new(max_sequence_length: usize, d_model: usize, theta: Option<f32>) -> Self {
        let theta = theta.unwrap_or(10000.0);
        RotaryEncodingConfig::new(max_sequence_length, d_model)
            .with_theta(theta)
            .init(&NDARRAYDEVICE)
            .into()
    }

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
implement_ndarray_interface!(
    RotaryEncodingRecordPy,
    RotaryEncodingRecord,
    "Record type of the RotaryEncoding layer."
);
implement_ndarray_interface!(
    SwiGluPy,
    SwiGlu,
    "Applies the SwiGLU or Swish Gated Linear Unit to the input tensor."
);

impl From<SwiGlu<NdArray>> for SwiGluPy {
    fn from(other: SwiGlu<NdArray>) -> Self {
        Self { inner: other }
    }
}

#[pymethods]
impl SwiGluPy {
    #[new]
    #[pyo3(signature = (d_input, d_output, bias = false, initializer = None))]
    fn new(
        d_input: usize,
        d_output: usize,
        bias: Option<bool>,
        initializer: Option<crate::nn::common_nn_exports::Initializer>,
    ) -> Self {
        let bias = bias.unwrap_or(false);
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
                crate::nn::common_nn_exports::Initializer::KaimingNormal { gain, fan_out_only } => {
                    Some(burn::nn::Initializer::KaimingNormal { gain, fan_out_only })
                }
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
            None => None,
        };
        match init {
            Some(init) => SwiGluConfig::new(d_input, d_output)
                .with_bias(bias)
                .with_initializer(init)
                .init(&NDARRAYDEVICE)
                .into(),

            None => SwiGluConfig::new(d_input, d_output)
                .with_bias(bias)
                .init(&NDARRAYDEVICE)
                .into(),
        }
    }

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

implement_send_and_sync!(SwiGluPy);
implement_send_and_sync!(RotaryEncodingPy);
implement_send_and_sync!(RotaryEncodingRecordPy);
implement_send_and_sync!(RmsNormPy);
implement_send_and_sync!(RmsNormRecordPy);
implement_send_and_sync!(PositionalEncodingRecordPy);
implement_send_and_sync!(PositionalEncodingPy);
implement_send_and_sync!(PReluRecordPy);
implement_send_and_sync!(PReluPy);
implement_send_and_sync!(LstmPy);
implement_send_and_sync!(LstmRecordPy);
implement_send_and_sync!(LayerNormPy);
implement_send_and_sync!(LayerNormRecordPy);
implement_send_and_sync!(InstanceNormRecordPy);
implement_send_and_sync!(InstanceNormPy);
implement_send_and_sync!(EmbeddingPy);
implement_send_and_sync!(GroupNormPy);
implement_send_and_sync!(GateControllerPy);

pub mod attention_exports {
    use super::*;
    use burn::nn::attention::*;

    implement_ndarray_interface!(
        GeneratePaddingMaskPy,
        GeneratePaddingMask,
        "Generate a padding attention mask."
    );

    impl From<GeneratePaddingMask<NdArray>> for GeneratePaddingMaskPy {
        fn from(other: GeneratePaddingMask<NdArray>) -> Self {
            Self { inner: other }
        }
    }

    #[pymethods]
    impl GeneratePaddingMaskPy {
        #[new]
        fn new(
            pad_token: usize,
            tokens_list: Vec<Vec<usize>>,
            max_seq_length: Option<usize>,
        ) -> Self {
            generate_padding_mask(pad_token, tokens_list, max_seq_length, &NDARRAYDEVICE).into()
        }
    }

    implement_ndarray_interface!(
        MhaCachePy,
        MhaCache,
        "Cache for the Multi Head Attention layer."
    );

    implement_ndarray_interface!(
        MhaInputPy,
        MhaInput,
        "Multihead attention forward pass input argument."
    );

    implement_ndarray_interface!(
        MultiHeadAttentionPy,
        MultiHeadAttention,
        "The multihead attention module as describe in the paper Attention Is All You Need."
    );

    implement_ndarray_interface!(MhaOutputPy, MhaOutput, "Multihead attention outputs.");

    implement_ndarray_interface!(
        MultiHeadAttentionRecordPy,
        MultiHeadAttentionRecord,
        "Record type for the MultiHeadAttention"
    );

    for_normal_struct_enums!(
        MultiHeadAttentionConfigPy,
        MultiHeadAttentionConfig,
        "Configuration for the MultiheadAttention module"
    );

    implement_send_and_sync!(MultiHeadAttentionRecordPy);
    implement_send_and_sync!(MultiHeadAttentionPy);
    implement_send_and_sync!(MhaOutputPy);
}

pub mod transformer_exports {
    use super::*;
    use burn::nn::transformer::*;

    implement_ndarray_interface!(
        PositionWiseFeedForwardRecordPy,
        PositionWiseFeedForwardRecord,
        "Record type for position wise feed forward record"
    );

    #[pyclass]
    pub struct PositionWiseFeedForwardPy {
        pub inner: PositionWiseFeedForward<NdArray>,
    }

    impl From<PositionWiseFeedForward<NdArray>> for PositionWiseFeedForwardPy {
        fn from(other: PositionWiseFeedForward<NdArray>) -> Self {
            Self { inner: other }
        }
    }

    #[pymethods]
    impl PositionWiseFeedForwardPy {
        /// Initializes a new PositionWiseFeedForward layer.
        ///
        /// params: d_model: The dimension of the input and output features.
        ///         d_ff: The dimension of the hidden inner features.
        ///         dropout: The dorpout rate. Defaults to 0.1
        ///        initializer: The weight initializer to use. Defaults to KaimingUniform. with gain of 1.0 and fan_out_only set to false.
        #[new]
        #[pyo3(signature = (d_model, d_ff, dropout = Some(0.1), initializer = None))]
        fn new(
            d_model: usize,
            d_ff: usize,
            dropout: Option<f64>,
            initializer: Option<crate::nn::common_nn_exports::Initializer>,
        ) -> Self {
            let dropout = dropout.unwrap_or(0.1);
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
                Some(init) => PositionWiseFeedForwardConfig::new(d_model, d_ff)
                    .with_dropout(dropout)
                    .with_initializer(init)
                    .init(&NDARRAYDEVICE)
                    .into(),
                None => PositionWiseFeedForwardConfig::new(d_model, d_ff)
                    .with_dropout(dropout)
                    .init(&NDARRAYDEVICE)
                    .into(),
            }
        }

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
    implement_ndarray_interface!(TransformerDecoderPy, TransformerDecoder);

    impl From<TransformerDecoder<NdArray>> for TransformerDecoderPy {
        fn from(other: TransformerDecoder<NdArray>) -> Self {
            Self { inner: other }
        }
    }

    // [TODO:] @kwach Implement all remaining methods of TransformerdecoderPy layer all that require the other layers for input
    //                  an autoregressive cache, forwarding method that performs autoregressive inference.
    #[pymethods]
    impl TransformerDecoderPy {
        /// Initializes a new TransformerDecoder layer.
        ///
        /// params: d_model: The size of the model
        ///         d_ff: The size of the positionwisefeedforward layer
        ///         n_heads: The number of attention heads
        ///         n_layers: The number of layers
        ///         dropout: The dropout rate. Defaults to 0.1
        ///         norm_first: Whether layer normalization will be applied first instead of after the other modules. Defaults to false.
        ///         quiet_softmax: Use “quiet softmax” instead of regular softmax.
        ///                             Usage may improve performance by allowing attention heads to deposit no information (if the sequence contains no information relevant to that head).
        ///                             Usage may reduce the entropy of weights in the model, enhancing quantization and compression
        ///         iniitializer: The weight initializer to use. Defaults to KaimingUniform. with gain of 1.0 and fan_out_only set to false.
        #[new]
        #[pyo3(signature = (d_model, d_ff, n_heads, n_layers, dropout = Some(0.1), norm_first = Some(false), quiet_softmax = Some(false), initializer = None))]
        fn new(
            d_model: usize,
            d_ff: usize,
            n_heads: usize,
            n_layers: usize,
            dropout: Option<f64>,
            norm_first: Option<bool>,
            quiet_softmax: Option<bool>,
            initializer: Option<crate::nn::common_nn_exports::Initializer>,
        ) -> Self {
            let dropout = dropout.unwrap_or(0.1);
            let norm_first = norm_first.unwrap_or(false);
            let quet_softmax = quiet_softmax.unwrap_or(false);
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
                Some(init) => TransformerDecoderConfig::new(d_model, d_ff, n_heads, n_layers)
                    .with_dropout(dropout)
                    .with_norm_first(norm_first)
                    .with_quiet_softmax(quet_softmax)
                    .with_initializer(init)
                    .init(&NDARRAYDEVICE)
                    .into(),
                None => TransformerDecoderConfig::new(d_model, d_ff, n_heads, n_layers)
                    .with_dropout(dropout)
                    .with_norm_first(norm_first)
                    .with_quiet_softmax(quet_softmax)
                    .init(&NDARRAYDEVICE)
                    .into(),
            }
        }

        // [TODO:] @kwach You need to test out these implementations in a Python setting; ie. the data may just be consumed and removed from memory

        fn forward(&self, input: &mut TransformerDecoderInputPy) -> TensorPy {
            let mut guard = input.inner.lock().unwrap().take().unwrap();
            self.inner.forward(guard).into()
            // match guard {
            //     Some(inner) => Ok(self.inner.forward(inner).into()),
            //     None => Err(TensorError::NonApplicableMethod.into()),
            // }
        }
    }
    implement_ndarray_interface!(
        TransformerDecoderAutoregressiveCachePy,
        TransformerDecoderAutoregressiveCache,
        "Autoregressive cache for the Transformer Decoder layer"
    );
    // implement_ndarray_interface!(
    //     TransformerDecoderInputPy,
    //     TransformerDecoderInput,
    //     "Transformer Decoder forward pass input argument"
    // );

    #[pyclass]
    pub struct TransformerDecoderInputPy {
        pub inner: Arc<Mutex<Option<TransformerDecoderInput<NdArray>>>>,
    }

    impl From<TransformerDecoderInput<NdArray>> for TransformerDecoderInputPy {
        fn from(other: TransformerDecoderInput<NdArray>) -> Self {
            Self {
                inner: Arc::new(Mutex::new(Some(other))),
            }
        }
    }

    #[pymethods]
    impl TransformerDecoderInputPy {
        #[new]
        fn new(target: TensorPy, memory: TensorPy) -> PyResult<Self> {
            match (target, memory) {
                (TensorPy::TensorThree(t1), TensorPy::TensorThree(t2)) => {
                    Ok(TransformerDecoderInput::new(t1.inner, t2.inner).into())
                }

                _ => Err(TensorError::NonApplicableMethod.into()),
            }
        }

        fn memory_mask_pad(&mut self, mask_pad: TensorPy) -> PyResult<Self> {
            match mask_pad {
                TensorPy::TensorTwoBool(tensor) => Ok(self
                    .inner
                    .lock()
                    .unwrap()
                    .take()
                    .unwrap()
                    .memory_mask_pad(tensor.inner)
                    .into()),
                _ => Err(TensorError::NonApplicableMethod.into()),
            }
        }
    }
    implement_ndarray_interface!(
        TransformerDecoderLayerPy,
        TransformerDecoderLayer,
        "Transformer Decoder layer module."
    );
    implement_ndarray_interface!(
        TransformerDecoderLayerRecordPy,
        TransformerDecoderLayerRecord,
        "Record type for the transformer decoder layer"
    );
    implement_ndarray_interface!(
        TransformerDecoderRecordPy,
        TransformerDecoderRecord,
        "Record type for the transformer decoder"
    );
    implement_ndarray_interface!(
        TransformerEncoderPy,
        TransformerEncoder,
        "The transformer encoder module as describe in the paper Attention Is All You Need."
    );

    impl From<TransformerEncoder<NdArray>> for TransformerEncoderPy {
        fn from(other: TransformerEncoder<NdArray>) -> Self {
            Self { inner: other }
        }
    }

    #[pymethods]
    impl TransformerEncoderPy {
        /// Initializes a new TransformerDecoder layer.
        ///
        /// params: d_model: The size of the model
        ///         d_ff: The size of the positionwisefeedforward layer
        ///         n_heads: The number of attention heads
        ///         n_layers: The number of layers
        ///         dropout: The dropout rate. Defaults to 0.1
        ///         norm_first: Whether layer normalization will be applied first instead of after the other modules. Defaults to false.
        ///         quiet_softmax: Use “quiet softmax” instead of regular softmax.
        ///                             Usage may improve performance by allowing attention heads to deposit no information (if the sequence contains no information relevant to that head).
        ///                             Usage may reduce the entropy of weights in the model, enhancing quantization and compression
        ///         iniitializer: The weight initializer to use. Defaults to KaimingUniform. with gain of 1.0 and fan_out_only set to false.
        #[new]
        #[pyo3(signature = (d_model, d_ff, n_heads, n_layers, dropout = Some(0.1), norm_first = Some(false), quiet_softmax = Some(false), initializer = None))]
        fn new(
            d_model: usize,
            d_ff: usize,
            n_heads: usize,
            n_layers: usize,
            dropout: Option<f64>,
            norm_first: Option<bool>,
            quiet_softmax: Option<bool>,
            initializer: Option<crate::nn::common_nn_exports::Initializer>,
        ) -> Self {
            let dropout = dropout.unwrap_or(0.1);
            let norm_first = norm_first.unwrap_or(false);
            let quet_softmax = quiet_softmax.unwrap_or(false);
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
                Some(init) => TransformerEncoderConfig::new(d_model, d_ff, n_heads, n_layers)
                    .with_dropout(dropout)
                    .with_norm_first(norm_first)
                    .with_quiet_softmax(quet_softmax)
                    .with_initializer(init)
                    .init(&NDARRAYDEVICE)
                    .into(),
                None => TransformerEncoderConfig::new(d_model, d_ff, n_heads, n_layers)
                    .with_dropout(dropout)
                    .with_norm_first(norm_first)
                    .with_quiet_softmax(quet_softmax)
                    .init(&NDARRAYDEVICE)
                    .into(),
            }
        }

        // [TODO:] @kwach You need to test out these implementations in a Python setting; ie. the data may just be consumed and removed from memory

        fn forward(&self, input: &mut TransformerEncoderInputPy) -> TensorPy {
            let mut guard = input.inner.lock().unwrap().take().unwrap();
            self.inner.forward(guard).into()
        }
    }
    implement_ndarray_interface!(
        TransformerEncoderAutoregressiveCachePy,
        TransformerEncoderAutoregressiveCache,
        "Autoregressive cache for the Transformer Encoder layer.\nTo be used during inference when decoding tokens."
    );
    implement_ndarray_interface!(
        TransformerEncoderLayerPy,
        TransformerEncoderLayer,
        "Transformer encoder layer module."
    );
    implement_ndarray_interface!(
        TransformerEncoderLayerRecordPy,
        TransformerEncoderLayerRecord,
        "Record type of the transformer encoder layer module"
    );
    implement_ndarray_interface!(
        TransformerEncoderRecordPy,
        TransformerEncoderRecord,
        "Record type of the transformer encoder module"
    );

    #[pyclass]
    pub struct TransformerEncoderInputPy {
        pub inner: Arc<Mutex<Option<TransformerEncoderInput<NdArray>>>>,
    }

    impl From<TransformerEncoderInput<NdArray>> for TransformerEncoderInputPy {
        fn from(other: TransformerEncoderInput<NdArray>) -> Self {
            Self {
                inner: Arc::new(Mutex::new(Some(other))),
            }
        }
    }

    #[pymethods]
    impl TransformerEncoderInputPy {
        #[new]
        fn new(tensor: TensorPy) -> PyResult<Self> {
            match (tensor) {
                (TensorPy::TensorThree(t1)) => Ok(TransformerEncoderInput::new(t1.inner).into()),

                _ => Err(TensorError::NonApplicableMethod.into()),
            }
        }

        fn mask_pad(&mut self, mask_pad: TensorPy) -> PyResult<Self> {
            match mask_pad {
                TensorPy::TensorTwoBool(tensor) => Ok(self
                    .inner
                    .lock()
                    .unwrap()
                    .take()
                    .unwrap()
                    .mask_pad(tensor.inner)
                    .into()),
                _ => Err(TensorError::NonApplicableMethod.into()),
            }
        }

        fn mask_attn(&mut self, mask_attn: TensorPy) -> PyResult<Self> {
            match mask_attn {
                TensorPy::TensorThreeBool(tensor) => Ok(self
                    .inner
                    .lock()
                    .unwrap()
                    .take()
                    .unwrap()
                    .mask_attn(tensor.inner)
                    .into()),
                _ => Err(TensorError::NonApplicableMethod.into()),
            }
        }
    }

    for_normal_struct_enums!(
        TransformerDecoderConfigPy,
        TransformerDecoderConfig,
        "Configuration to create a Transformer Decoder layer"
    );

    implement_send_and_sync!(TransformerEncoderRecordPy);
    implement_send_and_sync!(TransformerEncoderLayerRecordPy);
    implement_send_and_sync!(TransformerEncoderLayerPy);
    implement_send_and_sync!(TransformerEncoderInputPy);
    implement_send_and_sync!(TransformerEncoderAutoregressiveCachePy);
    implement_send_and_sync!(TransformerEncoderPy);
    implement_send_and_sync!(TransformerDecoderRecordPy);
    implement_send_and_sync!(TransformerDecoderLayerRecordPy);
    implement_send_and_sync!(TransformerDecoderLayerPy);
    implement_send_and_sync!(TransformerDecoderInputPy);
    implement_send_and_sync!(TransformerDecoderAutoregressiveCachePy);
    implement_send_and_sync!(TransformerDecoderPy);
    implement_send_and_sync!(PositionWiseFeedForwardPy);
    implement_send_and_sync!(PositionWiseFeedForwardRecordPy);
}

pub mod conv_exports {
    use super::*;
    use burn::nn::conv::*;
    use burn::prelude::*;

    implement_ndarray_interface!(
        DeformConv2dPy,
        DeformConv2d,
        "
Applies a deformable 2D convolution over input tensors."
    );

    impl From<DeformConv2d<NdArray>> for DeformConv2dPy {
        fn from(other: DeformConv2d<NdArray>) -> Self {
            Self { inner: other }
        }
    }

    #[pymethods]
    impl DeformConv2dPy {
        #[new]
        #[pyo3(signature = (channels, kernel_size, stride = None, dilation = None, weight_groups = None, offset_groups = None, padding = None, bias = Some(true), initializer = None))]
        fn new(
            channels: [usize; 2],
            kernel_size: [usize; 2],
            stride: Option<[usize; 2]>,
            dilation: Option<[usize; 2]>,
            weight_groups: Option<usize>,
            offset_groups: Option<usize>,
            padding: Option<PaddingConfig2dPy>,
            bias: Option<bool>,
            initializer: Option<crate::nn::common_nn_exports::Initializer>,
        ) -> Self {
            let stride = stride.unwrap_or([1, 1]);
            let offset_groups = offset_groups.unwrap_or(1);
            let dilation = dilation.unwrap_or([1, 1]);
            let weight_groups = weight_groups.unwrap_or(1);
            let padding = padding.unwrap_or(PaddingConfig2dPy::valid());
            let bias = bias.unwrap_or(true);
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
                None => None,
            };
            match init {
                Some(init) => DeformConv2dConfig::new(channels, kernel_size)
                    .with_stride(stride)
                    .with_dilation(dilation)
                    .with_weight_groups(weight_groups)
                    .with_offset_groups(offset_groups)
                    .with_padding(padding.0)
                    .with_bias(bias)
                    .with_initializer(init)
                    .init(&NDARRAYDEVICE)
                    .into(),
                None => DeformConv2dConfig::new(channels, kernel_size)
                    .with_stride(stride)
                    .with_dilation(dilation)
                    .with_weight_groups(weight_groups)
                    .with_offset_groups(offset_groups)
                    .with_padding(padding.0)
                    .with_bias(bias)
                    .init(&NDARRAYDEVICE)
                    .into(),
            }
        }

        fn forward(
            &self,
            input: TensorPy,
            offset: TensorPy,
            mask: Option<TensorPy>,
        ) -> PyResult<TensorPy> {
            match (input, offset, mask) {
                (
                    TensorPy::TensorFour(input_tensor),
                    TensorPy::TensorFour(offset_tensor),
                    Some(TensorPy::TensorFour(mask_tensor)),
                ) => Ok(self
                    .inner
                    .forward(
                        input_tensor.inner,
                        offset_tensor.inner,
                        Some(mask_tensor.inner),
                    )
                    .into()),
                (TensorPy::TensorFour(input_tensor), TensorPy::TensorFour(offset_tensor), None) => {
                    Ok(self
                        .inner
                        .forward(input_tensor.inner, offset_tensor.inner, None)
                        .into())
                }
                _ => Err(TensorError::NonApplicableMethod.into()),
            }
        }
    }

    implement_ndarray_interface!(
        DeformConv2dRecordPy,
        DeformConv2dRecord,
        "record type for the 2d deformable conolution module"
    );
    implement_ndarray_interface!(
        Conv1dPy,
        Conv1d,
        "Applies a 1D convolution over input tensors."
    );

    impl From<Conv1d<NdArray>> for Conv1dPy {
        fn from(other: Conv1d<NdArray>) -> Self {
            Self { inner: other }
        }
    }

    #[pymethods]
    impl Conv1dPy {
        #[new]
        #[pyo3(signature = (channels_in, channels_out, kernel_size, stride = Some(1), dilation = Some(1), groups = Some(1), padding = PaddingConfig1dPy::valid(), bias = Some(true), initializer = None))]
        fn new(
            channels_in: usize,
            channels_out: usize,
            kernel_size: usize,
            stride: Option<usize>,
            dilation: Option<usize>,
            groups: Option<usize>,
            padding: Option<PaddingConfig1dPy>,
            bias: Option<bool>,
            initializer: Option<crate::nn::common_nn_exports::Initializer>,
        ) -> Self {
            let stride = stride.unwrap_or(1);
            let dilation = dilation.unwrap_or(1);
            let groups = groups.unwrap_or(1);
            let bias = bias.unwrap_or(true);
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
                None => None,
            };
            let padding = padding.unwrap_or(PaddingConfig1dPy::valid());

            match init {
                Some(init) => {
                    burn::nn::conv::Conv1dConfig::new(channels_in, channels_out, kernel_size)
                        .with_stride(stride)
                        .with_dilation(dilation)
                        .with_padding(padding.0)
                        .with_bias(bias)
                        .with_initializer(init)
                        .init(&NDARRAYDEVICE)
                        .into()
                }
                None => burn::nn::conv::Conv1dConfig::new(channels_in, channels_out, kernel_size)
                    .with_stride(stride)
                    .with_dilation(dilation)
                    .with_padding(padding.0)
                    .with_bias(bias)
                    .init(&NDARRAYDEVICE)
                    .into(),
            }
        }

        fn foward(&self, input: TensorPy) -> PyResult<TensorPy> {
            match input {
                TensorPy::TensorThree(tensor) => Ok(self.inner.forward(tensor.inner).into()),
                _ => Err(TensorError::NonApplicableMethod.into()),
            }
        }
    }

    implement_ndarray_interface!(
        Conv1dRecordPy,
        Conv1dRecord,
        "record type for the 1D convolutional module."
    );
    implement_ndarray_interface!(
        Conv2dPy,
        Conv2d,
        "
Applies a 2D convolution over input tensors."
    );

    impl From<Conv2d<NdArray>> for Conv2dPy {
        fn from(other: Conv2d<NdArray>) -> Self {
            Self { inner: other }
        }
    }

    #[pymethods]
    impl Conv2dPy {
        #[new]
        #[pyo3(signature = (channels, kernel_size, dilation = Some([1,1]) , stride = Some([1,1]), groups = Some(1), padding = None, bias = Some(true), initializer = None))]
        fn new(
            channels: [usize; 2],
            kernel_size: [usize; 2],
            dilation: Option<[usize; 2]>,
            stride: Option<[usize; 2]>,
            groups: Option<usize>,
            padding: Option<PaddingConfig2dPy>,
            bias: Option<bool>,
            initializer: Option<crate::nn::common_nn_exports::Initializer>,
        ) -> Self {
            let stride = stride.unwrap_or([1, 1]);
            let dilation = dilation.unwrap_or([1, 1]);
            let groups = groups.unwrap_or(1);
            let bias = bias.unwrap_or(true);
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
                None => None,
            };
            let padding = padding.unwrap_or(PaddingConfig2dPy::valid());

            match init {
                Some(init) => burn::nn::conv::Conv2dConfig::new(channels, kernel_size)
                    .with_stride(stride)
                    .with_dilation(dilation)
                    .with_padding(padding.0)
                    .with_groups(groups)
                    .with_bias(bias)
                    .with_initializer(init)
                    .init(&NDARRAYDEVICE)
                    .into(),
                None => burn::nn::conv::Conv2dConfig::new(channels, kernel_size)
                    .with_stride(stride)
                    .with_dilation(dilation)
                    .with_padding(padding.0)
                    .with_groups(groups)
                    .with_bias(bias)
                    .init(&NDARRAYDEVICE)
                    .into(),
            }
        }

        fn forward(&self, input: TensorPy) -> PyResult<TensorPy> {
            match input {
                TensorPy::TensorFour(tensor) => Ok(self.inner.forward(tensor.inner).into()),

                _ => Err(TensorError::NonApplicableMethod.into()),
            }
        }
    }
    implement_ndarray_interface!(
        Conv2dRecordPy,
        Conv2dRecord,
        "record type for the 2D convolutional module."
    );
    implement_ndarray_interface!(
        Conv3DPy,
        Conv3d,
        "
Applies a 3D convolution over input tensors."
    );

    impl From<Conv3d<NdArray>> for Conv3DPy {
        fn from(other: Conv3d<NdArray>) -> Self {
            Self { inner: other }
        }
    }

    #[pymethods]
    impl Conv3DPy {
        #[new]
        #[pyo3(signature = (channels, kernel_size, dilation = Some([1,1,1]) , stride = Some([1,1,1]), groups = Some(1), padding = None, bias = Some(true), initializer = None))]
        fn new(
            channels: [usize; 2],
            kernel_size: [usize; 3],
            dilation: Option<[usize; 3]>,
            stride: Option<[usize; 3]>,
            groups: Option<usize>,
            padding: Option<PaddingConfig3dPy>,
            bias: Option<bool>,
            initializer: Option<crate::nn::common_nn_exports::Initializer>,
        ) -> Self {
            let stride = stride.unwrap_or([1, 1, 1]);
            let dilation = dilation.unwrap_or([1, 1, 1]);
            let groups = groups.unwrap_or(1);
            let bias = bias.unwrap_or(true);
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
                None => None,
            };
            let padding = padding.unwrap_or(PaddingConfig3dPy::valid());

            match init {
                Some(init) => burn::nn::conv::Conv3dConfig::new(channels, kernel_size)
                    .with_stride(stride)
                    .with_dilation(dilation)
                    .with_padding(padding.0)
                    .with_groups(groups)
                    .with_bias(bias)
                    .with_initializer(init)
                    .init(&NDARRAYDEVICE)
                    .into(),
                None => burn::nn::conv::Conv3dConfig::new(channels, kernel_size)
                    .with_stride(stride)
                    .with_dilation(dilation)
                    .with_padding(padding.0)
                    .with_groups(groups)
                    .with_bias(bias)
                    .init(&NDARRAYDEVICE)
                    .into(),
            }
        }

        fn forward(&self, input: TensorPy) -> PyResult<TensorPy> {
            match input {
                TensorPy::TensorFive(tensor) => Ok(self.inner.forward(tensor.inner).into()),

                _ => Err(TensorError::NonApplicableMethod.into()),
            }
        }
    }
    implement_ndarray_interface!(
        ConvTranspose1dPy,
        ConvTranspose1d,
        "Applies a 1D transposed convolution over input tensors"
    );

    impl From<ConvTranspose1d<NdArray>> for ConvTranspose1dPy {
        fn from(other: ConvTranspose1d<NdArray>) -> Self {
            Self { inner: other }
        }
    }

    #[pymethods]
    impl ConvTranspose1dPy {
        #[new]
        #[pyo3(signature = (channels ,kernel_size, stride = Some(1), dilation = Some(1), groups = Some(1), padding = 0, bias = Some(true), initializer = None))]
        fn new(
            channels: [usize; 2],
            kernel_size: usize,
            stride: Option<usize>,
            dilation: Option<usize>,
            groups: Option<usize>,
            padding: Option<usize>,
            bias: Option<bool>,
            initializer: Option<crate::nn::common_nn_exports::Initializer>,
        ) -> Self {
            let stride = stride.unwrap_or(1);
            let dilation = dilation.unwrap_or(1);
            let groups = groups.unwrap_or(1);
            let bias = bias.unwrap_or(true);
            let padding = padding.unwrap_or(0);
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
                None => None,
            };

            match init {
                Some(init) => burn::nn::conv::ConvTranspose1dConfig::new(channels, kernel_size)
                    .with_stride(stride)
                    .with_dilation(dilation)
                    .with_padding(padding)
                    .with_groups(groups)
                    .with_bias(bias)
                    .with_initializer(init)
                    .init(&NDARRAYDEVICE)
                    .into(),
                None => burn::nn::conv::ConvTranspose1dConfig::new(channels, kernel_size)
                    .with_stride(stride)
                    .with_dilation(dilation)
                    .with_padding(padding)
                    .with_groups(groups)
                    .with_bias(bias)
                    .init(&NDARRAYDEVICE)
                    .into(),
            }
        }

        fn forward(&self, input: TensorPy) -> PyResult<TensorPy> {
            match input {
                TensorPy::TensorThree(tensor) => Ok(self.inner.forward(tensor.inner).into()),
                _ => Err(TensorError::NonApplicableMethod.into()),
            }
        }
    }

    implement_ndarray_interface!(
        ConvTranspose1dRecordPy,
        ConvTranspose1dRecord,
        " record type for the 1D convolutional transpose module."
    );
    implement_ndarray_interface!(
        ConvTranspose2dPy,
        ConvTranspose2d,
        "Applies a 2D transposed convolution over input tensors."
    );

    impl From<ConvTranspose2d<NdArray>> for ConvTranspose2dPy {
        fn from(other: ConvTranspose2d<NdArray>) -> Self {
            Self { inner: other }
        }
    }

    #[pymethods]
    impl ConvTranspose2dPy {
        #[new]
        #[pyo3(signature = (channels ,kernel_size, stride = Some([1,1]), dilation = Some([1,1]), groups = Some(1), padding = [0,0], padding_out = [0,0] , bias = Some(true), initializer = None))]
        fn new(
            channels: [usize; 2],
            kernel_size: [usize; 2],
            stride: Option<[usize; 2]>,
            dilation: Option<[usize; 2]>,
            groups: Option<usize>,
            padding: Option<[usize; 2]>,
            padding_out: Option<[usize; 2]>,
            bias: Option<bool>,
            initializer: Option<crate::nn::common_nn_exports::Initializer>,
        ) -> Self {
            let stride = stride.unwrap_or([1, 1]);
            let dilation = dilation.unwrap_or([1, 1]);
            let groups = groups.unwrap_or(1);
            let bias = bias.unwrap_or(true);
            let padding_out = padding_out.unwrap_or([0, 0]);
            let padding = padding.unwrap_or([0, 0]);
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
                None => None,
            };

            match init {
                Some(init) => burn::nn::conv::ConvTranspose2dConfig::new(channels, kernel_size)
                    .with_stride(stride)
                    .with_dilation(dilation)
                    .with_padding(padding)
                    .with_groups(groups)
                    .with_bias(bias)
                    .with_initializer(init)
                    .init(&NDARRAYDEVICE)
                    .into(),
                None => burn::nn::conv::ConvTranspose2dConfig::new(channels, kernel_size)
                    .with_stride(stride)
                    .with_dilation(dilation)
                    .with_padding(padding)
                    .with_groups(groups)
                    .with_bias(bias)
                    .init(&NDARRAYDEVICE)
                    .into(),
            }
        }

        fn forward(&self, input: TensorPy) -> PyResult<TensorPy> {
            match input {
                TensorPy::TensorFour(tensor) => Ok(self.inner.forward(tensor.inner).into()),
                _ => Err(TensorError::NonApplicableMethod.into()),
            }
        }
    }

    implement_ndarray_interface!(
        ConvTranspose2dRecordPy,
        ConvTranspose2dRecord,
        "record type for the 3D convolutional transpose module"
    );
    implement_ndarray_interface!(
        ConvTranspose3dPy,
        ConvTranspose3d,
        "Applies a 3D transposed convolution over input tensors."
    );

    implement_ndarray_interface!(
        ConvTranspose3dRecordPy,
        ConvTranspose3dRecord,
        " record type for the 3D convolutional transpose module."
    );

    impl From<ConvTranspose3d<NdArray>> for ConvTranspose3dPy {
        fn from(other: ConvTranspose3d<NdArray>) -> Self {
            Self { inner: other }
        }
    }

    #[pymethods]
    impl ConvTranspose3dPy {
        #[new]
        #[pyo3(signature = (channels ,kernel_size, stride = Some([1,1,1]), dilation = Some([1,1,1]), groups = Some(1), padding = [0,0,0], padding_out = [0,0,0] , bias = Some(true), initializer = None))]
        fn new(
            channels: [usize; 2],
            kernel_size: [usize; 3],
            stride: Option<[usize; 3]>,
            dilation: Option<[usize; 3]>,
            groups: Option<usize>,
            padding: Option<[usize; 3]>,
            padding_out: Option<[usize; 3]>,
            bias: Option<bool>,
            initializer: Option<crate::nn::common_nn_exports::Initializer>,
        ) -> Self {
            let stride = stride.unwrap_or([1, 1, 1]);
            let dilation = dilation.unwrap_or([1, 1, 1]);
            let groups = groups.unwrap_or(1);
            let bias = bias.unwrap_or(true);
            let padding_out = padding_out.unwrap_or([0, 0, 0]);
            let padding = padding.unwrap_or([0, 0, 0]);
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
                None => None,
            };

            match init {
                Some(init) => burn::nn::conv::ConvTranspose3dConfig::new(channels, kernel_size)
                    .with_stride(stride)
                    .with_dilation(dilation)
                    .with_padding(padding)
                    .with_groups(groups)
                    .with_bias(bias)
                    .with_initializer(init)
                    .init(&NDARRAYDEVICE)
                    .into(),
                None => burn::nn::conv::ConvTranspose3dConfig::new(channels, kernel_size)
                    .with_stride(stride)
                    .with_dilation(dilation)
                    .with_padding(padding)
                    .with_groups(groups)
                    .with_bias(bias)
                    .init(&NDARRAYDEVICE)
                    .into(),
            }
        }

        fn forward(&self, input: TensorPy) -> PyResult<TensorPy> {
            match input {
                TensorPy::TensorFive(tensor) => Ok(self.inner.forward(tensor.inner).into()),
                _ => Err(TensorError::NonApplicableMethod.into()),
            }
        }
    }

    implement_send_and_sync!(Conv1dPy);
    implement_send_and_sync!(Conv3DPy);
    implement_send_and_sync!(Conv1dRecordPy);
    implement_send_and_sync!(Conv2dPy);
    implement_send_and_sync!(Conv2dRecordPy);
    implement_send_and_sync!(ConvTranspose1dPy);
    implement_send_and_sync!(ConvTranspose1dRecordPy);
    implement_send_and_sync!(ConvTranspose2dPy);
    implement_send_and_sync!(ConvTranspose2dRecordPy);
    implement_send_and_sync!(ConvTranspose3dPy);
    implement_send_and_sync!(ConvTranspose3dRecordPy);
    implement_send_and_sync!(DeformConv2dPy);
    implement_send_and_sync!(DeformConv2dRecordPy);
}

pub mod gru_exports {
    use super::*;
    use burn::nn::gru::*;

    implement_ndarray_interface!(GruRecordPy, GruRecord, "record type for the Gru module");
    implement_ndarray_interface!(
        GruPy,
        Gru,
        "The Gru (Gated recurrent unit) module. This implementation is for a unidirectional, stateless, Gru."
    );

    impl From<Gru<NdArray>> for GruPy {
        fn from(other: Gru<NdArray>) -> Self {
            Self { inner: other }
        }
    }

    #[pymethods]
    impl GruPy {
        /// To instantiate a new gru module.
        ///
        /// Default values:
        /// - bias: True
        /// - reset_after: True
        /// - initializer: Initializer::KaimingUniform { gain: 1.0, fan_out_only: false }
        #[new]
        #[pyo3(signature = (d_input, d_hidden, bias, reset_after = true, initializer = None))]
        fn new(
            d_input: usize,
            d_hidden: usize,
            bias: bool,
            reset_after: Option<bool>,
            initializer: Option<crate::nn::common_nn_exports::Initializer>,
        ) -> Self {
            // let bias = bias.unwrap_or(true);
            let reset_after = reset_after.unwrap_or(true);
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
                None => None,
            };
            match init {
                Some(init) => GruConfig::new(d_input, d_hidden, bias)
                    .with_reset_after(reset_after)
                    .with_initializer(init)
                    .init(&NDARRAYDEVICE)
                    .into(),
                None => GruConfig::new(d_input, d_hidden, bias)
                    .with_reset_after(reset_after)
                    .init(&NDARRAYDEVICE)
                    .into(),
            }
        }

        fn forward(&self, batched_input: TensorPy, state: Option<TensorPy>) -> PyResult<TensorPy> {
            match (batched_input, state) {
                (
                    TensorPy::TensorThree(batched_input_tensor),
                    Some(TensorPy::TensorTwo(state_tensor)),
                ) => Ok(self
                    .inner
                    .forward(batched_input_tensor.inner, Some(state_tensor.inner))
                    .into()),
                (TensorPy::TensorThree(batched_input_tensor), None) => {
                    Ok(self.inner.forward(batched_input_tensor.inner, None).into())
                }
                _ => Err(TensorError::NonApplicableMethod.into()),
            }
        }
    }

    implement_send_and_sync!(GruRecordPy);
    implement_send_and_sync!(GruPy);
}
