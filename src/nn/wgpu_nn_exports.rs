use std::usize;

use crate::{for_normal_struct_enums, implement_send_and_sync, implement_wgpu_interface};
use burn::nn::*;
use burn::prelude::*;
use pyo3::prelude::*;
use pyo3::types::*;
use burn::backend::Wgpu;

#[derive(Clone)]
#[pyclass]
pub struct Tensor1 {
    pub inner: Tensor<Wgpu, 1>
}

#[derive(Clone)]
#[pyclass]
pub struct Tensor2 {
    pub inner: Tensor<Wgpu, 2>
}

#[derive(Clone)]
#[pyclass]
pub struct Tensor3 {
    pub inner: Tensor<Wgpu, 3>
}

#[derive(Clone)]
#[pyclass]
pub struct Tensor4 {
    pub inner: Tensor<Wgpu, 4>
}

#[derive(Clone)]
#[pyclass]
pub struct Tensor5 {
    pub inner: Tensor<Wgpu, 5>
}

#[pyclass]
#[derive(Clone)]
pub enum PyTensor {
    TensorOne(Tensor1),
    TensorTwo(Tensor2),
    TensorThree(Tensor3),
    TensorFour(Tensor4),
    TensorFive(Tensor5)
}

impl From<Tensor<Wgpu,1>> for Tensor1 {
    fn from(other: Tensor<Wgpu,1>)-> Self {
        Self{inner: other}
    }
}

impl From<Tensor<Wgpu, 1>> for PyTensor {
    fn from(other: Tensor<Wgpu, 1>)-> Self {
        Self::TensorOne(other.into())
    }
}

impl From<Tensor<Wgpu,2>> for Tensor2 {
    fn from(other: Tensor<Wgpu,2>)-> Self {
        Self{inner: other}
    }
}

impl From<Tensor<Wgpu,2>> for PyTensor {
    fn from(other: Tensor<Wgpu,2>)-> Self {
        Self::TensorTwo(other.into())
    }
}

impl From<Tensor<Wgpu,3>> for Tensor3 {
    fn from(other: Tensor<Wgpu,3>)-> Self {
        Self{inner: other}
    }
}

impl From<Tensor<Wgpu,3>> for PyTensor {
    fn from(other: Tensor<Wgpu,3>)-> Self {
        Self::TensorThree(other.into())
    }
}

impl From<Tensor<Wgpu,4>> for Tensor4 {
    fn from(other: Tensor<Wgpu,4>)-> Self {
        Self{inner: other}
    }
}

impl From<Tensor<Wgpu,4>> for PyTensor {
    fn from(other: Tensor<Wgpu,4>)-> Self {
        Self::TensorFour(other.into())
    }
}

impl From<Tensor<Wgpu,5>> for Tensor5 {
    fn from(other: Tensor<Wgpu,5>)-> Self {
        Self{inner: other}
    }
}

impl From<Tensor<Wgpu,5>> for PyTensor {
    fn from(other: Tensor<Wgpu,5>)-> Self {
        Self::TensorFive(other.into())
    }
}


// These methods appear to be totally redundant but anyway
impl From<Tensor1> for Tensor<Wgpu, 1> {
    fn from(other: Tensor1) -> Self {
        other.inner
    }
}

impl From<Tensor2> for Tensor<Wgpu, 2> {
    fn from(other: Tensor2) -> Self {
        other.inner
    }
}

impl From<Tensor3> for Tensor<Wgpu, 3> {
    fn from(other: Tensor3) -> Self {
        other.inner
    }
}

impl From<Tensor4> for Tensor<Wgpu, 4> {
    fn from(other: Tensor4) -> Self {
        other.inner
    }
}

impl From<Tensor5> for Tensor<Wgpu, 5> {
    fn from(other: Tensor5) -> Self {
        other.inner
    }
}


// Conversions from PyTensor into its internal types
// impl From<PyTensor> for Tensor<Wgpu,3> {
//     fn from(other: PyTensor) -> Self {
//         match other {
//             PyTensor::TensorThree(val) => val.into(),
//             _ => 
//         }
//     }
// }

// [`TODO`] Update the documentation to reference the papers. Some of us learn through these frameworks.
implement_wgpu_interface!(
    PyGateController,
    GateController,
    "A GateController represents a gate in an LSTM cell.\n An LSTM cell generally contains three gates: an input gate, forget gate,\n and output gate. Additionally, cell gate is just used to compute the cell state"
);

    
implement_wgpu_interface!(
    PyEmbedding,
    Embedding,
    "Lookup table to store a fix number of vectors."
);

implement_wgpu_interface!(
    PyGroupNorm,
    GroupNorm,
    "Applies Group Normalization over a mini-batch of inputs"
);



implement_wgpu_interface!(
    PyInstanceNorm,
    InstanceNorm,
    "Applies Instance Normalization over a tensor"
);

implement_wgpu_interface!(
    PyInstanceNormRecord,
    InstanceNormRecord,
    "Record type of the InstanceNorm module"
);


implement_wgpu_interface!(
    PyLayerNorm,
    LayerNorm,
    "Applies Layer Normalization over a tensor"
);


implement_wgpu_interface!(
    PyLayerNormRecord,
    LayerNormRecord,
    "Record type of the LayerNorm record"
);


// implement_wgpu_interface!(PyLinearRecord, LinearRecord);
implement_wgpu_interface!(
    PyLstm,
    Lstm,
    "The Lstm module. This implementation is for a unidirectional, stateless, Lstm"
);


implement_wgpu_interface!(PyLstmRecord, LstmRecord, "Record type of the Lstm module");
implement_wgpu_interface!(PyPRelu, PRelu, "Parametric Relu Layer");
implement_wgpu_interface!(
    PyPReluRecord,
    PReluRecord,
    "record type of the PRelu module"
);

implement_wgpu_interface!(PyPositionalEncoding, PositionalEncoding, "
Positional encoding layer for transformer models \n This layer adds positional information to the input embeddings,\nallowing the transformer model to take into account the order of the sequence.\n The positional encoding is added to the input embeddings by computing\n a set of sinusoidal functions with different frequencies and phases.");

implement_wgpu_interface!(
    PyPositionalEncodingRecord,
    PositionalEncodingRecord,
    "Record type of the PositionalEncoding module"
);


implement_wgpu_interface!(
    PyRmsNorm,
    RmsNorm,
    "Applies RMS Normalization over an input tensor along the last dimension"
);

implement_wgpu_interface!(
    PyRmsNormRecord,
    RmsNormRecord,
    "Record type of the RmsNormRecord"
);

implement_wgpu_interface!(
    PyRotaryEncoding,
    RotaryEncoding,
    "A module that applies rotary positional encoding to a tensor.\n Rotary Position Encoding or Embedding (RoPE), is a type of \nposition embedding which encodes absolute positional\n information with rotation matrix and naturally incorporates explicit relative \nposition dependency in self-attention formulation."
);




implement_wgpu_interface!(
    PyRotaryEncodingRecord,
    RotaryEncodingRecord,
    "Record type of the RotaryEncoding layer."
);



implement_wgpu_interface!(
    PySwiGlu,
    SwiGlu,
    "Applies the SwiGLU or Swish Gated Linear Unit to the input tensor."
);

// implement_wgpu_interface!(PySwiGluRecord, SwiGluRecord);

for_normal_struct_enums!(PyUnfold4d, Unfold4d, "Four-dimensional unfolding.");

for_normal_struct_enums!(
    PyUnfold4dConfig,
    Unfold4dConfig,
    "Configuration to create unfold4d layer"
);

for_normal_struct_enums!(
    PyTanh,
    Tanh,
    "Applies the tanh activation function element-wise"
);

for_normal_struct_enums!(
    PySwiGluConfig,
    SwiGluConfig,
    "Configuration to create a SwiGlu activation layer"
);


for_normal_struct_enums!(
    PyPositionalEncodingConfig,
    PositionalEncodingConfig,
    "Configuration to create a PositionalEncoding layer"
);
for_normal_struct_enums!(
    PyPReluConfig,
    PReluConfig,
    "Configuration to create the PRelu layer"
);
for_normal_struct_enums!(
    PyLstmConfig,
    LstmConfig,
    "Configuration to create a Lstm module"
);
for_normal_struct_enums!(PyLeakyRelu, LeakyRelu, "LeakyRelu Layer");
for_normal_struct_enums!(
    PyLeakyReluConfig,
    LeakyReluConfig,
    "Configuration to create the LeakyRelu layer"
);
for_normal_struct_enums!(
    PyGeLu,
    Gelu,
    "Applies the Gaussian Error Linear Units function element-wise."
);
for_normal_struct_enums!(PyHardSigmoid, HardSigmoid, "HardSigmoid Layer");
for_normal_struct_enums!(
    PyHardSigmoidConfig,
    HardSigmoidConfig,
    "Configuration to build the HardSigmoid layer"
);
for_normal_struct_enums!(
    PyInstanceNormConfig,
    InstanceNormConfig,
    "Configuration to create a InstanceNorm layer"
);
for_normal_struct_enums!(
    PyLayerNormConfig,
    LayerNormConfig,
    "Configuration to create a LayerNorm layer "
);
for_normal_struct_enums!(
    PyRmsNormConfig,
    RmsNormConfig,
    "Configuration to create a RMS Norm layer"
);
for_normal_struct_enums!(
    PySigmoid,
    Sigmoid,
    "Applies the sigmoid function element-wise"
);
for_normal_struct_enums!(
    PyInitializer,
    Initializer,
    "Enum specifying with what values a tensor should be initialized"
);

// [TODO*] There are methods exposed by this type that are relevant for uploading config files for 
// reproduction of train/test results
for_normal_struct_enums!(
    PyPaddingConfig1d,
    PaddingConfig1d,
    "Padding configuration for 1D operators.
    With three options: Same, Valid and Explicit
    * Same - Dynamically calculate the amount of padding necessary to ensure that the output size will be the same as the input.
    * Valid - Same as no padding
    * Explicit - Takes an input and applies the specified amount of padding to all inputs."
);


#[pymethods]
impl PyPaddingConfig1d {
    #[classattr]
    pub fn same() -> Self {
        PyPaddingConfig1d(PaddingConfig1d::Same)
    }

    #[classattr]
    pub fn valid() -> Self {
        PyPaddingConfig1d(PaddingConfig1d::Valid)
    }

    #[staticmethod]
    pub fn explicit(val: usize) -> Self {
        PyPaddingConfig1d(PaddingConfig1d::Explicit(val))
    }
}

for_normal_struct_enums!(
    PyPaddingConfig2d,
    PaddingConfig2d,
    "Padding configuration for 2D operators."
);

#[pymethods]
impl PyPaddingConfig2d {
    #[classattr]
    pub fn same() -> Self {
        PyPaddingConfig2d(PaddingConfig2d::Same)
    }

    #[classattr]
    pub fn valid() -> Self {
        PyPaddingConfig2d(PaddingConfig2d::Valid)
    }

    #[staticmethod]
    pub fn explicit(val1: usize, val2: usize) -> Self {
        PyPaddingConfig2d(PaddingConfig2d::Explicit(val1, val2))
    }
}


for_normal_struct_enums!(
    PyPaddingConfig3d,
    PaddingConfig3d,
    "Padding configuration for 3D operators."
);

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
implement_send_and_sync!(PyGateController);

pub mod attention_exports {
    use super::*;
    use burn::nn::attention::*;

    // vec![GeneratePaddingMask, MhaCache, MhaInput, MultiHeadAttention];

    implement_wgpu_interface!(
        PyGeneratePaddingMask,
        GeneratePaddingMask,
        "Generate a padding attention mask."
    );
    implement_wgpu_interface!(
        PyMhaCache,
        MhaCache,
        "Cache for the Multi Head Attention layer."
    );
    implement_wgpu_interface!(
        PyMhaInput,
        MhaInput,
        "Multihead attention forward pass input argument."
    );
    implement_wgpu_interface!(
        PyMultiHeadAttention,
        MultiHeadAttention,
        "The multihead attention module as describe in the paper Attention Is All You Need."
    );
    implement_wgpu_interface!(PyMhaOutput, MhaOutput, "Multihead attention outputs.");
    implement_wgpu_interface!(
        PyMultiHeadAttentionRecord,
        MultiHeadAttentionRecord,
        "Record type for the MultiHeadAttention"
    );

    for_normal_struct_enums!(
        PyMultiHeadAttentionConfig,
        MultiHeadAttentionConfig,
        "Configuration for the MultiheadAttention module"
    );

    implement_send_and_sync!(PyMultiHeadAttentionRecord);
    implement_send_and_sync!(PyMultiHeadAttention);
    implement_send_and_sync!(PyMhaOutput);
}

pub mod transformer_exports {
    use super::*;
    use burn::nn::transformer::*;

    implement_wgpu_interface!(
        PyPositionWiseFeedForwardRecord,
        PositionWiseFeedForwardRecord,
        "Record type for position wise feed forward record"
    );

    implement_wgpu_interface!(PyTransformerDecoder, TransformerDecoder);
    implement_wgpu_interface!(
        PyTransformerDecoderAutoregressiveCache,
        TransformerDecoderAutoregressiveCache,
        "Autoregressive cache for the Transformer Decoder layer"
    );
    implement_wgpu_interface!(
        PyTransformerDecoderInput,
        TransformerDecoderInput,
        "Transformer Decoder forward pass input argument"
    );
    implement_wgpu_interface!(
        PyTransformerDecoderLayer,
        TransformerDecoderLayer,
        "Transformer Decoder layer module."
    );
    implement_wgpu_interface!(
        PyTransformerDecoderLayerRecord,
        TransformerDecoderLayerRecord,
        "Record type for the transformer decoder layer"
    );
    implement_wgpu_interface!(
        PyTransformerDecoderRecord,
        TransformerDecoderRecord,
        "Record type for the transformer decoder"
    );
    implement_wgpu_interface!(
        PyTransformerEncoder,
        TransformerEncoder,
        "The transformer encoder module as describe in the paper Attention Is All You Need."
    );
    implement_wgpu_interface!(
        PyTransformerEncoderAutoregressiveCache,
        TransformerEncoderAutoregressiveCache,
        "Autoregressive cache for the Transformer Encoder layer.\nTo be used during inference when decoding tokens."
    );
    implement_wgpu_interface!(
        PyTransformerEncoderLayer,
        TransformerEncoderLayer,
        "Transformer encoder layer module."
    );
    implement_wgpu_interface!(
        PyTransformerEncoderLayerRecord,
        TransformerEncoderLayerRecord,
        "Record type of the transformer encoder layer module"
    );
    implement_wgpu_interface!(
        PyTransformerEncoderRecord,
        TransformerEncoderRecord,
        "Record type of the transformer encoder module"
    );
    implement_wgpu_interface!(
        PyTransformerEncoderInput,
        TransformerEncoderInput,
        "Transformer Encoder forward pass input argument"
    );

    for_normal_struct_enums!(
        PyPositionWiseFeedForwardConfig,
        PositionWiseFeedForwardConfig,
        "Configuration to create a position-wise feed-forward layer"
    );
    for_normal_struct_enums!(
        PyTransformerDecoderConfig,
        TransformerDecoderConfig,
        "Configuration to create a Transformer Decoder layer"
    );

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
    implement_send_and_sync!(PyPositionWiseFeedForwardRecord);
}

pub mod conv_exports {
    use super::*;
    use burn::nn::conv::*;
    use burn::prelude::*;

    implement_wgpu_interface!(
        PyDeformConv2d,
        DeformConv2d,
        "
Applies a deformable 2D convolution over input tensors."
    );
    implement_wgpu_interface!(
        PyDeformConv2dRecord,
        DeformConv2dRecord,
        "record type for the 2d deformable conolution module"
    );
    implement_wgpu_interface!(
        PyConv1d,
        Conv1d,
        "Applies a 1D convolution over input tensors."
    );
    implement_wgpu_interface!(
        PyConv1dRecord,
        Conv1dRecord,
        "record type for the 1D convolutional module."
    );
    implement_wgpu_interface!(
        PyConv2d,
        Conv2d,
        "
Applies a 2D convolution over input tensors."
    );
    implement_wgpu_interface!(
        PyConv2dRecord,
        Conv2dRecord,
        "record type for the 2D convolutional module."
    );
    implement_wgpu_interface!(
        PyConv3D,
        Conv3d,
        "
Applies a 3D convolution over input tensors."
    );
    implement_wgpu_interface!(
        PyConvTranspose1d,
        ConvTranspose1d,
        "Applies a 1D transposed convolution over input tensors"
    );
    implement_wgpu_interface!(
        PyConvTranspose1dRecord,
        ConvTranspose1dRecord,
        " record type for the 1D convolutional transpose module."
    );
    implement_wgpu_interface!(
        PyConvTranspose2d,
        ConvTranspose2d,
        "Applies a 2D transposed convolution over input tensors."
    );
    implement_wgpu_interface!(
        PyConvTranspose2dRecord,
        ConvTranspose2dRecord,
        "record type for the 3D convolutional transpose module"
    );
    implement_wgpu_interface!(
        PyConvTranspose3d,
        ConvTranspose3d,
        "Applies a 3D transposed convolution over input tensors."
    );
    implement_wgpu_interface!(
        PyConvTranspose3dRecord,
        ConvTranspose3dRecord,
        " record type for the 3D convolutional transpose module."
    );

    for_normal_struct_enums!(
        PyDeformConv2dConfig,
        DeformConv2dConfig,
        "Configuration for the 2d Deform convolution layer."
    );
    for_normal_struct_enums!(
        PyConvTranspose1dConfig,
        ConvTranspose1dConfig,
        "Configuration to create a 1D convolution transpose layer."
    );
    for_normal_struct_enums!(
        PyConvTranspose2dConfig,
        ConvTranspose2dConfig,
        "Configuration to create a 2D convolution transpose layer"
    );
    for_normal_struct_enums!(
        PyConvTranspose3dConfig,
        ConvTranspose3dConfig,
        "Configuration to create a 3D convolution transpose layer"
    );
    for_normal_struct_enums!(
        PyConv1DConfig,
        Conv1dConfig,
        "Configuration to create a 1D convolution layer"
    );
    for_normal_struct_enums!(
        PyConv2DConfig,
        Conv2dConfig,
        "
Configuration to create a 2D convolution layer,"
    );
    for_normal_struct_enums!(
        PyConv3DConfig,
        Conv3dConfig,
        "
Configuration to create a 3D convolution layer,"
    );

    implement_send_and_sync!(PyConv1d);
    implement_send_and_sync!(PyConv3D);
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

pub mod gru_exports {
    use super::*;
    use burn::nn::gru::*;

    implement_wgpu_interface!(PyGruRecord, GruRecord, "record type for the Gru module");
    implement_wgpu_interface!(PyGru, Gru, "
The Gru (Gated recurrent unit) module. This implementation is for a unidirectional, stateless, Gru.");

    for_normal_struct_enums!(
        PyGruConfig,
        GruConfig,
        "
Configuration to create a gru module"
    );

    implement_send_and_sync!(PyGruRecord);
    implement_send_and_sync!(PyGru);
}
