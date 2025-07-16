use crate::{
    for_normal_struct_enums, implement_send_and_sync,
    implement_wgpu_interface,
};
use burn::nn::Linear;
use burn::nn::*;
use burn::prelude::*;
use pyo3::prelude::*;

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
for_normal_struct_enums!(
    PyPaddingConfig1d,
    PaddingConfig1d,
    "Padding configuration for 1D operators."
);
for_normal_struct_enums!(
    PyPaddingConfig2d,
    PaddingConfig2d,
    "Padding configuration for 2D operators."
);
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