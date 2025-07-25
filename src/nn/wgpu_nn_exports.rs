use std::ops::DerefMut;
use std::usize;

use crate::tensor::tensor_error::TensorError;
use crate::{for_normal_struct_enums, implement_send_and_sync, implement_wgpu_interface};
use burn::backend::Wgpu;
use burn::nn::*;
use burn::prelude::*;
use pyo3::prelude::*;
use pyo3::types::*;

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
    GateControllerPy,
    GateController,
    "A GateController represents a gate in an LSTM cell.\n An LSTM cell generally contains three gates: an input gate, forget gate,\n and output gate. Additionally, cell gate is just used to compute the cell state"
);

implement_wgpu_interface!(
    EmbeddingPy,
    Embedding,
    "Lookup table to store a fix number of vectors."
);

implement_wgpu_interface!(
    GroupNormPy,
    GroupNorm,
    "Applies Group Normalization over a mini-batch of inputs"
);

implement_wgpu_interface!(
    InstanceNormPy,
    InstanceNorm,
    "Applies Instance Normalization over a tensor"
);

implement_wgpu_interface!(
    InstanceNormRecordPy,
    InstanceNormRecord,
    "Record type of the InstanceNorm module"
);

implement_wgpu_interface!(
    LayerNormPy,
    LayerNorm,
    "Applies Layer Normalization over a tensor"
);

implement_wgpu_interface!(
    LayerNormRecordPy,
    LayerNormRecord,
    "Record type of the LayerNorm record"
);

// implement_wgpu_interface!(PyLinearRecord, LinearRecord);
implement_wgpu_interface!(
    LstmPy,
    Lstm,
    "The Lstm module. This implementation is for a unidirectional, stateless, Lstm"
);

implement_wgpu_interface!(LstmRecordPy, LstmRecord, "Record type of the Lstm module");
implement_wgpu_interface!(PReluPy, PRelu, "Parametric Relu Layer");
implement_wgpu_interface!(
    PReluRecordPy,
    PReluRecord,
    "record type of the PRelu module"
);

implement_wgpu_interface!(PositionalEncodingPy, PositionalEncoding, "
Positional encoding layer for transformer models \n This layer adds positional information to the input embeddings,\nallowing the transformer model to take into account the order of the sequence.\n The positional encoding is added to the input embeddings by computing\n a set of sinusoidal functions with different frequencies and phases.");

implement_wgpu_interface!(
    PositionalEncodingRecordPy,
    PositionalEncodingRecord,
    "Record type of the PositionalEncoding module"
);

implement_wgpu_interface!(
    RmsNormPy,
    RmsNorm,
    "Applies RMS Normalization over an input tensor along the last dimension"
);

implement_wgpu_interface!(
    RmsNormRecordPy,
    RmsNormRecord,
    "Record type of the RmsNormRecord"
);

implement_wgpu_interface!(
    RotaryEncodingPy,
    RotaryEncoding,
    "A module that applies rotary positional encoding to a tensor.\n Rotary Position Encoding or Embedding (RoPE), is a type of \nposition embedding which encodes absolute positional\n information with rotation matrix and naturally incorporates explicit relative \nposition dependency in self-attention formulation."
);

implement_wgpu_interface!(
    RotaryEncodingRecordPy,
    RotaryEncodingRecord,
    "Record type of the RotaryEncoding layer."
);

implement_wgpu_interface!(
    SwiGluPy,
    SwiGlu,
    "Applies the SwiGLU or Swish Gated Linear Unit to the input tensor."
);

// implement_wgpu_interface!(PySwiGluRecord, SwiGluRecord);

for_normal_struct_enums!(Unfold4dPy, Unfold4d, "Four-dimensional unfolding.");

for_normal_struct_enums!(
    Unfold4dConfigPy,
    Unfold4dConfig,
    "Configuration to create unfold4d layer"
);

for_normal_struct_enums!(
    TanhPy,
    Tanh,
    "Applies the tanh activation function element-wise"
);

for_normal_struct_enums!(
    SwiGluConfigPy,
    SwiGluConfig,
    "Configuration to create a SwiGlu activation layer"
);

for_normal_struct_enums!(
    PositionalEncodingConfigPy,
    PositionalEncodingConfig,
    "Configuration to create a PositionalEncoding layer"
);
for_normal_struct_enums!(
    PReluConfigPy,
    PReluConfig,
    "Configuration to create the PRelu layer"
);
for_normal_struct_enums!(
    LstmConfigPy,
    LstmConfig,
    "Configuration to create a Lstm module"
);
for_normal_struct_enums!(LeakyReluPy, LeakyRelu, "LeakyRelu Layer");
for_normal_struct_enums!(
    LeakyReluConfigPy,
    LeakyReluConfig,
    "Configuration to create the LeakyRelu layer"
);
for_normal_struct_enums!(
    GeLuPy,
    Gelu,
    "Applies the Gaussian Error Linear Units function element-wise."
);
for_normal_struct_enums!(HardSigmoidPy, HardSigmoid, "HardSigmoid Layer");
for_normal_struct_enums!(
    HardSigmoidConfigPy,
    HardSigmoidConfig,
    "Configuration to build the HardSigmoid layer"
);
for_normal_struct_enums!(
    InstanceNormConfigPy,
    InstanceNormConfig,
    "Configuration to create a InstanceNorm layer"
);
for_normal_struct_enums!(
    LayerNormConfigPy,
    LayerNormConfig,
    "Configuration to create a LayerNorm layer "
);
for_normal_struct_enums!(
    RmsNormConfigPy,
    RmsNormConfig,
    "Configuration to create a RMS Norm layer"
);
for_normal_struct_enums!(
    SigmoidPy,
    Sigmoid,
    "Applies the sigmoid function element-wise"
);
for_normal_struct_enums!(
    InitializerPy,
    Initializer,
    "Enum specifying with what values a tensor should be initialized"
);

// [TODO*] There are methods exposed by this type that are relevant for uploading config files for
// reproduction of train/test results
for_normal_struct_enums!(
    PaddingConfig1dPy,
    PaddingConfig1d,
    "Padding configuration for 1D operators.
    With three options: Same, Valid and Explicit
    * Same - Dynamically calculate the amount of padding necessary to ensure that the output size will be the same as the input.
    * Valid - Same as no padding
    * Explicit - Takes an input and applies the specified amount of padding to all inputs."
);

#[pymethods]
impl PaddingConfig1dPy {
    #[classattr]
    pub fn same() -> Self {
        PaddingConfig1dPy(PaddingConfig1d::Same)
    }

    #[classattr]
    pub fn valid() -> Self {
        PaddingConfig1dPy(PaddingConfig1d::Valid)
    }

    #[staticmethod]
    pub fn explicit(val: usize) -> Self {
        PaddingConfig1dPy(PaddingConfig1d::Explicit(val))
    }
}

for_normal_struct_enums!(
    PaddingConfig2dPy,
    PaddingConfig2d,
    "Padding configuration for 2D operators."
);

#[pymethods]
impl PaddingConfig2dPy {
    #[classattr]
    pub fn same() -> Self {
        PaddingConfig2dPy(PaddingConfig2d::Same)
    }

    #[classattr]
    pub fn valid() -> Self {
        PaddingConfig2dPy(PaddingConfig2d::Valid)
    }

    #[staticmethod]
    pub fn explicit(val1: usize, val2: usize) -> Self {
        PaddingConfig2dPy(PaddingConfig2d::Explicit(val1, val2))
    }
}

for_normal_struct_enums!(
    PaddingConfig3dPy,
    PaddingConfig3d,
    "Padding configuration for 3D operators."
);

implement_send_and_sync!(SwiGluPy);
// implement_send_and_sync!(PySwiGluRecord);
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
// implement_send_and_sync!(PyLinearRecord);
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

    // vec![GeneratePaddingMask, MhaCache, MhaInput, MultiHeadAttention];

    implement_wgpu_interface!(
        GeneratePaddingMaskPy,
        GeneratePaddingMask,
        "Generate a padding attention mask."
    );
    implement_wgpu_interface!(
        MhaCachePy,
        MhaCache,
        "Cache for the Multi Head Attention layer."
    );
    implement_wgpu_interface!(
        MhaInputPy,
        MhaInput,
        "Multihead attention forward pass input argument."
    );
    implement_wgpu_interface!(
        MultiHeadAttentionPy,
        MultiHeadAttention,
        "The multihead attention module as describe in the paper Attention Is All You Need."
    );
    implement_wgpu_interface!(MhaOutputPy, MhaOutput, "Multihead attention outputs.");
    implement_wgpu_interface!(
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

    implement_wgpu_interface!(
        PositionWiseFeedForwardRecordPy,
        PositionWiseFeedForwardRecord,
        "Record type for position wise feed forward record"
    );

    implement_wgpu_interface!(TransformerDecoderPy, TransformerDecoder);
    implement_wgpu_interface!(
        TransformerDecoderAutoregressiveCachePy,
        TransformerDecoderAutoregressiveCache,
        "Autoregressive cache for the Transformer Decoder layer"
    );
    implement_wgpu_interface!(
        TransformerDecoderInputPy,
        TransformerDecoderInput,
        "Transformer Decoder forward pass input argument"
    );
    implement_wgpu_interface!(
        TransformerDecoderLayerPy,
        TransformerDecoderLayer,
        "Transformer Decoder layer module."
    );
    implement_wgpu_interface!(
        TransformerDecoderLayerRecordPy,
        TransformerDecoderLayerRecord,
        "Record type for the transformer decoder layer"
    );
    implement_wgpu_interface!(
        TransformerDecoderRecordPy,
        TransformerDecoderRecord,
        "Record type for the transformer decoder"
    );
    implement_wgpu_interface!(
        TransformerEncoderPy,
        TransformerEncoder,
        "The transformer encoder module as describe in the paper Attention Is All You Need."
    );
    implement_wgpu_interface!(
        TransformerEncoderAutoregressiveCachePy,
        TransformerEncoderAutoregressiveCache,
        "Autoregressive cache for the Transformer Encoder layer.\nTo be used during inference when decoding tokens."
    );
    implement_wgpu_interface!(
        TransformerEncoderLayerPy,
        TransformerEncoderLayer,
        "Transformer encoder layer module."
    );
    implement_wgpu_interface!(
        TransformerEncoderLayerRecordPy,
        TransformerEncoderLayerRecord,
        "Record type of the transformer encoder layer module"
    );
    implement_wgpu_interface!(
        TransformerEncoderRecordPy,
        TransformerEncoderRecord,
        "Record type of the transformer encoder module"
    );
    implement_wgpu_interface!(
        TransformerEncoderInputPy,
        TransformerEncoderInput,
        "Transformer Encoder forward pass input argument"
    );

    for_normal_struct_enums!(
        PositionWiseFeedForwardConfigPy,
        PositionWiseFeedForwardConfig,
        "Configuration to create a position-wise feed-forward layer"
    );
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
    implement_send_and_sync!(PositionWiseFeedForwardRecordPy);
}

pub mod conv_exports {
    use super::*;
    use burn::nn::conv::*;
    use burn::prelude::*;

    implement_wgpu_interface!(
        DeformConv2dPy,
        DeformConv2d,
        "
Applies a deformable 2D convolution over input tensors."
    );
    implement_wgpu_interface!(
        DeformConv2dRecordPy,
        DeformConv2dRecord,
        "record type for the 2d deformable conolution module"
    );
    implement_wgpu_interface!(
        Conv1dPy,
        Conv1d,
        "Applies a 1D convolution over input tensors."
    );
    implement_wgpu_interface!(
        Conv1dRecordPy,
        Conv1dRecord,
        "record type for the 1D convolutional module."
    );
    implement_wgpu_interface!(
        Conv2dPy,
        Conv2d,
        "
Applies a 2D convolution over input tensors."
    );
    implement_wgpu_interface!(
        Conv2dRecordPy,
        Conv2dRecord,
        "record type for the 2D convolutional module."
    );
    implement_wgpu_interface!(
        Conv3DPy,
        Conv3d,
        "
Applies a 3D convolution over input tensors."
    );
    implement_wgpu_interface!(
        ConvTranspose1dPy,
        ConvTranspose1d,
        "Applies a 1D transposed convolution over input tensors"
    );
    implement_wgpu_interface!(
        ConvTranspose1dRecordPy,
        ConvTranspose1dRecord,
        " record type for the 1D convolutional transpose module."
    );
    implement_wgpu_interface!(
        ConvTranspose2dPy,
        ConvTranspose2d,
        "Applies a 2D transposed convolution over input tensors."
    );
    implement_wgpu_interface!(
        ConvTranspose2dRecordPy,
        ConvTranspose2dRecord,
        "record type for the 3D convolutional transpose module"
    );
    implement_wgpu_interface!(
        ConvTranspose3dPy,
        ConvTranspose3d,
        "Applies a 3D transposed convolution over input tensors."
    );
    implement_wgpu_interface!(
        ConvTranspose3dRecordPy,
        ConvTranspose3dRecord,
        " record type for the 3D convolutional transpose module."
    );

    for_normal_struct_enums!(
        DeformConv2dConfigPy,
        DeformConv2dConfig,
        "Configuration for the 2d Deform convolution layer."
    );
    for_normal_struct_enums!(
        ConvTranspose1dConfigPy,
        ConvTranspose1dConfig,
        "Configuration to create a 1D convolution transpose layer."
    );
    for_normal_struct_enums!(
        ConvTranspose2dConfigPy,
        ConvTranspose2dConfig,
        "Configuration to create a 2D convolution transpose layer"
    );
    for_normal_struct_enums!(
        ConvTranspose3dConfigPy,
        ConvTranspose3dConfig,
        "Configuration to create a 3D convolution transpose layer"
    );
    for_normal_struct_enums!(
        Conv1DConfigPy,
        Conv1dConfig,
        "Configuration to create a 1D convolution layer"
    );
    for_normal_struct_enums!(
        Conv2DConfigPy,
        Conv2dConfig,
        "
Configuration to create a 2D convolution layer,"
    );
    for_normal_struct_enums!(
        Conv3DConfigPy,
        Conv3dConfig,
        "
Configuration to create a 3D convolution layer,"
    );

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

    implement_wgpu_interface!(GruRecordPy, GruRecord, "record type for the Gru module");
    implement_wgpu_interface!(GruPy, Gru, "
The Gru (Gated recurrent unit) module. This implementation is for a unidirectional, stateless, Gru.");

    for_normal_struct_enums!(
        GruConfigPy,
        GruConfig,
        "
Configuration to create a gru module"
    );

    implement_send_and_sync!(GruRecordPy);
    implement_send_and_sync!(GruPy);
}
