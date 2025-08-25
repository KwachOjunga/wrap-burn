use super::wgpu_nn_exports::*;
use crate::for_normal_struct_enums;
use crate::tensor::tensor_error::TensorError;
use crate::tensor::wgpu_base::*;
use burn::nn::*;
use burn::prelude::*;
use pyo3::prelude::*;
use pyo3::pyclass_init::PyClassInitializer;
use pyo3::types::PyInt;

/// Enum specifying with what values a tensor should be initialized
#[pyclass]
#[derive(Clone, Debug)]
pub enum Initializer {
    Constant { value: f64 },
    One(),
    Zero(),
    Uniform { min: f64, max: f64 },
    Normal { mean: f64, std: f64 },
    KaimingUniform { gain: f64, fan_out_only: bool },
    KaimingNormal { gain: f64, fan_out_only: bool },
    XavierUniform { gain: f64 },
    XavierNormal { gain: f64 },
    Orthogonal { gain: f64 },
}

for_normal_struct_enums!(Unfold4dPy, Unfold4d, "Four-dimensional unfolding.");

#[pymethods]
impl Unfold4dPy {
    #[new]
    #[pyo3(signature = (kernel_size, stride = [1,1], dilation = [1,1], padding = [0,0]))]
    fn new(
        kernel_size: [usize; 2],
        stride: Option<[usize; 2]>,
        dilation: Option<[usize; 2]>,
        padding: Option<[usize; 2]>,
    ) -> Self {
        let stride = stride.unwrap_or([1, 1]);
        let dilation = dilation.unwrap_or([1, 1]);
        let padding = padding.unwrap_or([0, 0]);

        Unfold4dConfig::new(kernel_size)
            .with_stride(stride)
            .with_dilation(dilation)
            .with_padding(padding)
            .init()
            .into()
    }

    fn forward(&self, input: TensorPy) -> PyResult<TensorPy> {
        match input {
            TensorPy::TensorFour(tensor) => Ok(self.0.forward(tensor.inner).into()),
            _ => Err(TensorError::NonApplicableMethod.into()),
        }
    }
}

for_normal_struct_enums!(
    TanhPy,
    Tanh,
    "Applies the tanh activation function element-wise"
);

#[pymethods]
impl TanhPy {
    #[new]
    fn new() -> Self {
        TanhPy(Tanh::new())
    }

    fn forward(&self, input: TensorPy) -> PyResult<TensorPy> {
        match input {
            TensorPy::TensorOne(tensor) => Ok(self.0.forward(tensor.inner).into()),
            TensorPy::TensorTwo(tensor) => Ok(self.0.forward(tensor.inner).into()),
            TensorPy::TensorThree(tensor) => Ok(self.0.forward(tensor.inner).into()),
            TensorPy::TensorFour(tensor) => Ok(self.0.forward(tensor.inner).into()),
            TensorPy::TensorFive(tensor) => Ok(self.0.forward(tensor.inner).into()),
            _ => Err(TensorError::NonApplicableMethod.into()),
        }
    }
}

for_normal_struct_enums!(LeakyReluPy, LeakyRelu, "LeakyRelu Layer");

#[pymethods]
impl LeakyReluPy {
    #[new]
    #[pyo3(signature = (negative_slope = None))]
    fn new(negative_slope: Option<f64>) -> Self {
        match negative_slope {
            Some(slope) => LeakyReluConfig::new()
                .with_negative_slope(slope)
                .init()
                .into(),
            None => LeakyReluConfig::new().init().into(),
        }
    }

    fn forward(&self, input: TensorPy) -> PyResult<TensorPy> {
        match input {
            TensorPy::TensorOne(tensor) => Ok(self.0.forward(tensor.inner).into()),
            TensorPy::TensorTwo(tensor) => Ok(self.0.forward(tensor.inner).into()),
            TensorPy::TensorThree(tensor) => Ok(self.0.forward(tensor.inner).into()),
            TensorPy::TensorFour(tensor) => Ok(self.0.forward(tensor.inner).into()),
            TensorPy::TensorFive(tensor) => Ok(self.0.forward(tensor.inner).into()),
            _ => Err(TensorError::NonApplicableMethod.into()),
        }
    }
}

for_normal_struct_enums!(
    GeLuPy,
    Gelu,
    "Applies the Gaussian Error Linear Units function element-wise."
);

#[pymethods]
impl GeLuPy {
    #[new]
    fn new() -> Self {
        Gelu::new().into()
    }

    fn forward(&self, input: TensorPy) -> PyResult<TensorPy> {
        match input {
            TensorPy::TensorOne(tensor) => Ok(self.0.forward(tensor.inner).into()),
            TensorPy::TensorTwo(tensor) => Ok(self.0.forward(tensor.inner).into()),
            TensorPy::TensorThree(tensor) => Ok(self.0.forward(tensor.inner).into()),
            TensorPy::TensorFour(tensor) => Ok(self.0.forward(tensor.inner).into()),
            TensorPy::TensorFive(tensor) => Ok(self.0.forward(tensor.inner).into()),
            _ => Err(TensorError::NonApplicableMethod.into()),
        }
    }
}

for_normal_struct_enums!(HardSigmoidPy, HardSigmoid, "HardSigmoid Layer");

#[pymethods]
impl HardSigmoidPy {
    #[new]
    #[pyo3(signature = (alpha = Some(0.2), beta = Some(0.5)))]
    fn new(alpha: Option<f64>, beta: Option<f64>) -> Self {
        let alpha = alpha.unwrap_or(0.2);
        let beta = beta.unwrap_or(0.5);
        HardSigmoidConfig::new()
            .with_alpha(alpha)
            .with_beta(beta)
            .init()
            .into()
    }

    fn forward(&self, input: TensorPy) -> PyResult<TensorPy> {
        match input {
            TensorPy::TensorOne(tensor) => Ok(self.0.forward(tensor.inner).into()),
            TensorPy::TensorTwo(tensor) => Ok(self.0.forward(tensor.inner).into()),
            TensorPy::TensorThree(tensor) => Ok(self.0.forward(tensor.inner).into()),
            TensorPy::TensorFour(tensor) => Ok(self.0.forward(tensor.inner).into()),
            TensorPy::TensorFive(tensor) => Ok(self.0.forward(tensor.inner).into()),
            _ => Err(TensorError::NonApplicableMethod.into()),
        }
    }
}

for_normal_struct_enums!(
    SigmoidPy,
    Sigmoid,
    "Applies the sigmoid function element-wise"
);

#[pymethods]
impl SigmoidPy {
    #[new]
    fn new() -> Self {
        Sigmoid::new().into()
    }

    fn forward(&self, input: TensorPy) -> PyResult<TensorPy> {
        match input {
            TensorPy::TensorOne(tensor) => Ok(self.0.forward(tensor.inner).into()),
            TensorPy::TensorTwo(tensor) => Ok(self.0.forward(tensor.inner).into()),
            TensorPy::TensorThree(tensor) => Ok(self.0.forward(tensor.inner).into()),
            TensorPy::TensorFour(tensor) => Ok(self.0.forward(tensor.inner).into()),
            TensorPy::TensorFive(tensor) => Ok(self.0.forward(tensor.inner).into()),
            _ => Err(TensorError::NonApplicableMethod.into()),
        }
    }
}

// for_normal_struct_enums!(
//     InitializerPy,
//     Initializer,
//     "Enum specifying with what values a tensor should be initialized"
// );

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

#[pymethods]
impl PaddingConfig3dPy {
    #[classattr]
    pub fn same() -> Self {
        PaddingConfig3dPy(PaddingConfig3d::Same)
    }

    #[classattr]
    pub fn valid() -> Self {
        PaddingConfig3dPy(PaddingConfig3d::Valid)
    }

    #[staticmethod]
    pub fn explicit(val1: usize, val2: usize, val3: usize) -> Self {
        PaddingConfig3dPy(PaddingConfig3d::Explicit(val1, val2, val3))
    }
}

// [TODO] @kwach, implement Initializer methods to produce Tensors.

pub mod pool_exports {
    pub(crate) use super::*;
    use burn::{backend::Wgpu, nn::pool::*};
    use pyo3::exceptions::PyResourceWarning;

    /// This is  the typical AdaptivePool1d layer
    for_normal_struct_enums!(
        AdaptiveAvgPool1dPy,
        AdaptiveAvgPool1d,
        "Applies a 1D adaptive avg pooling over input tensors"
    );

    for_normal_struct_enums!(
        AdaptiveAvgPool1dConfigPy,
        AdaptiveAvgPool1dConfig,
        "Configuration to create a 1D adaptive avg pooling layer"
    );

    for_normal_struct_enums!(
        AdaptiveAvgPool2dPy,
        AdaptiveAvgPool2d,
        "Applies a 2D adaptive avg pooling over input tensors"
    );
    for_normal_struct_enums!(
        AdaptiveAvgPool2dConfigPy,
        AdaptiveAvgPool2dConfig,
        "Configuration to create a 2D adaptive avg pooling layer"
    );
    for_normal_struct_enums!(
        AvgPool1dPy,
        AvgPool1d,
        "Applies a 1D avg pooling over input tensors."
    );
    for_normal_struct_enums!(
        AvgPool1dConfigPy,
        AvgPool1dConfig,
        "
Configuration to create a 1D avg pooling layer"
    );
    for_normal_struct_enums!(
        AvgPool2dPy,
        AvgPool2d,
        "Applies a 2D avg pooling over input tensors."
    );
    for_normal_struct_enums!(
        AvgPool2dConfigPy,
        AvgPool2dConfig,
        "Configuration to create a 2D avg pooling layer"
    );
    for_normal_struct_enums!(
        MaxPool1dPy,
        MaxPool1d,
        "Applies a 1D max pooling over input tensors."
    );
    for_normal_struct_enums!(
        MaxPool1dConfigPy,
        MaxPool1dConfig,
        "
Configuration to create a 1D max pooling layer"
    );
    for_normal_struct_enums!(
        MaxPool2dPy,
        MaxPool2d,
        "
Applies a 2D max pooling over input tensors."
    );
    for_normal_struct_enums!(
        MaxPool2dConfigPy,
        MaxPool2dConfig,
        "Configuration to create a 2D max pooling layer "
    );

    // Methods section
    // PyAdaptivePool1d

    #[pymethods]
    impl AdaptiveAvgPool1dPy {
        #[getter]
        fn output(&self) -> PyResult<usize> {
            Ok(self.0.output_size)
        }

        #[new]
        fn new(output: usize) -> Self {
            let mut pool_layer = AdaptiveAvgPool1dConfig::new(output);
            pool_layer.init().into()
        }

        /// Perform a feedforward tensor operation on a 3 dimensional tensor
        fn forward(&self, tensor: TensorPy) -> PyResult<TensorPy> {
            match tensor {
                TensorPy::TensorThree(val) => Ok(self.0.forward(val.inner).into()),
                _ => Err(TensorError::WrongDimensions.into()),
            }
        }
    }

    //[NOTE**] PyAdaptiveAvgPool2d

    #[pymethods]
    impl AdaptiveAvgPool2dPy {
        #[getter]
        fn output(&self) -> PyResult<[usize; 2]> {
            Ok(self.0.output_size)
        }

        #[new]
        fn new(output: [usize; 2]) -> Self {
            let mut pool_layer = AdaptiveAvgPool2dConfig::new(output);
            pool_layer.init().into()
        }

        /// Perform a feedforward tensor operation on a 3 dimensional tensor
        fn forward(&self, tensor: TensorPy) -> PyResult<TensorPy> {
            match tensor {
                TensorPy::TensorFour(val) => Ok(self.0.forward(val.inner).into()),
                _ => Err(TensorError::WrongDimensions.into()),
            }
        }
    }

    // [NOTE**] PyAvgPool1d
    #[pymethods]
    impl AvgPool1dPy {
        // #[classmethod]
        #[new]
        #[pyo3(signature = (kernel_size , stride = None, padding = None, count_bool_pad = None))]
        fn new(
            py: Python<'_>,
            kernel_size: usize,
            stride: Option<usize>,
            padding: Option<PaddingConfig1dPy>,
            count_bool_pad: Option<bool>,
        ) -> AvgPool1dPy {
            let stride = stride.unwrap_or(1);
            let padding = padding.unwrap_or(PaddingConfig1dPy::valid());
            let count_bool_pad = count_bool_pad.unwrap_or(true);

            AvgPool1dConfigPy::new(kernel_size)
                .with_stride(py, stride)
                .with_padding(py, padding)
                .with_count_include_pad(count_bool_pad)
                .init()
        }
    }

    #[pymethods]
    impl AvgPool1dConfigPy {
        #[staticmethod]
        pub fn new(kernel_size: usize) -> AvgPool1dConfigPy {
            AvgPool1dConfigPy(AvgPool1dConfig::new(kernel_size))
        }

        pub fn with_stride(&self, py: Python<'_>, stride: usize) -> AvgPool1dConfigPy {
            AvgPool1dConfigPy(self.0.clone().with_stride(stride))
        }

        pub fn with_padding(
            &mut self,
            py: Python<'_>,
            padding: PaddingConfig1dPy,
        ) -> AvgPool1dConfigPy {
            match padding.0 {
                PaddingConfig1d::Same => {
                    AvgPool1dConfigPy(self.0.clone().with_padding(PaddingConfig1d::Same))
                }
                PaddingConfig1d::Valid => {
                    AvgPool1dConfigPy(self.0.clone().with_padding(PaddingConfig1d::Valid))
                }
                PaddingConfig1d::Explicit(val) => {
                    AvgPool1dConfigPy(self.0.clone().with_padding(PaddingConfig1d::Explicit(val)))
                }
            }
        }

        pub fn with_count_include_pad(&self, pad: bool) -> AvgPool1dConfigPy {
            AvgPool1dConfigPy(self.0.clone().with_count_include_pad(pad))
        }

        fn init(&self) -> AvgPool1dPy {
            AvgPool1dPy(self.0.init())
        }
    }

    //[NOTE**] PyAvgPool2d

    #[pymethods]
    impl AvgPool2dPy {
        // #[classmethod]
        #[new]
        #[pyo3(signature = (kernel_size , stride = None, padding = None, count_bool_pad = None))]
        fn new(
            py: Python<'_>,
            kernel_size: [usize; 2],
            stride: Option<[usize; 2]>,
            padding: Option<PaddingConfig2dPy>,
            count_bool_pad: Option<bool>,
        ) -> AvgPool2dPy {
            let stride = stride.unwrap_or([1, 1]);
            let padding = padding.unwrap_or(PaddingConfig2dPy::valid());
            let count_bool_pad = count_bool_pad.unwrap_or(true);

            AvgPool2dConfigPy::new(kernel_size)
                .with_strides(py, stride)
                .with_padding(py, padding)
                .with_count_include_pad(count_bool_pad)
                .init()
        }
    }

    #[pymethods]
    impl AvgPool2dConfigPy {
        #[new]
        pub fn new(kernel_size: [usize; 2]) -> AvgPool2dConfigPy {
            AvgPool2dConfigPy(AvgPool2dConfig::new(kernel_size))
        }

        pub fn with_strides(&self, py: Python<'_>, stride: [usize; 2]) -> AvgPool2dConfigPy {
            AvgPool2dConfigPy(self.0.clone().with_strides(stride))
        }

        pub fn with_padding(
            &mut self,
            py: Python<'_>,
            padding: PaddingConfig2dPy,
        ) -> AvgPool2dConfigPy {
            match padding.0 {
                PaddingConfig2d::Same => {
                    AvgPool2dConfigPy(self.0.clone().with_padding(PaddingConfig2d::Same))
                }
                PaddingConfig2d::Valid => {
                    AvgPool2dConfigPy(self.0.clone().with_padding(PaddingConfig2d::Valid))
                }
                PaddingConfig2d::Explicit(val1, val2) => AvgPool2dConfigPy(
                    self.0
                        .clone()
                        .with_padding(PaddingConfig2d::Explicit(val1, val2)),
                ),
            }
        }

        pub fn with_count_include_pad(&self, pad: bool) -> AvgPool2dConfigPy {
            AvgPool2dConfigPy(self.0.clone().with_count_include_pad(pad))
        }

        fn init(&self) -> AvgPool2dPy {
            AvgPool2dPy(self.0.init())
        }
    }

    //[NOTE**] PyMaxPool1d

    #[pymethods]
    impl MaxPool1dPy {
        // #[classmethod]
        #[staticmethod]
        #[pyo3(signature = (kernel_size , stride = None, padding = None, dilation = Some(1)))]
        fn new(
            py: Python<'_>,
            kernel_size: usize,
            stride: Option<usize>,
            padding: Option<PaddingConfig1dPy>,
            dilation: Option<usize>,
        ) -> MaxPool1dPy {
            let stride = stride.unwrap_or(1);
            let padding = padding.unwrap_or(PaddingConfig1dPy::valid());
            let dilation = dilation.unwrap_or(1);

            MaxPool1dConfigPy::new(kernel_size)
                .with_stride(py, stride)
                .with_padding(py, padding)
                .with_dilation(dilation)
                .init()
        }
    }

    #[pymethods]
    impl MaxPool1dConfigPy {
        #[staticmethod]
        pub fn new(kernel_size: usize) -> MaxPool1dConfigPy {
            MaxPool1dConfigPy(MaxPool1dConfig::new(kernel_size))
        }

        pub fn with_stride(&self, py: Python<'_>, stride: usize) -> MaxPool1dConfigPy {
            MaxPool1dConfigPy(self.0.clone().with_stride(stride))
        }

        pub fn with_padding(
            &mut self,
            py: Python<'_>,
            padding: PaddingConfig1dPy,
        ) -> MaxPool1dConfigPy {
            match padding.0 {
                PaddingConfig1d::Same => {
                    MaxPool1dConfigPy(self.0.clone().with_padding(PaddingConfig1d::Same))
                }
                PaddingConfig1d::Valid => {
                    MaxPool1dConfigPy(self.0.clone().with_padding(PaddingConfig1d::Valid))
                }
                PaddingConfig1d::Explicit(val) => {
                    MaxPool1dConfigPy(self.0.clone().with_padding(PaddingConfig1d::Explicit(val)))
                }
            }
        }

        pub fn with_dilation(&self, dilation: usize) -> MaxPool1dConfigPy {
            MaxPool1dConfigPy(self.0.clone().with_dilation(dilation))
        }

        fn init(&self) -> MaxPool1dPy {
            MaxPool1dPy(self.0.init())
        }
    }

    // [NOTE**] PyMaxPool2d

    #[pymethods]
    impl MaxPool2dPy {
        // #[classmethod]
        #[staticmethod]
        #[pyo3(signature = (kernel_size , stride = None, padding = None, dilation = None))]
        fn new(
            py: Python<'_>,
            kernel_size: [usize; 2],
            stride: Option<[usize; 2]>,
            padding: Option<PaddingConfig2dPy>,
            dilation: Option<[usize; 2]>,
        ) -> MaxPool2dPy {
            let stride = stride.unwrap_or([1, 1]);
            let padding = padding.unwrap_or(PaddingConfig2dPy::valid());
            let dilation = dilation.unwrap_or([1, 1]);

            MaxPool2dConfigPy::new(kernel_size)
                .with_strides(py, stride)
                .with_padding(py, padding)
                .with_dilation(dilation)
                .init()
        }
    }

    #[pymethods]
    impl MaxPool2dConfigPy {
        #[staticmethod]
        pub fn new(kernel_size: [usize; 2]) -> MaxPool2dConfigPy {
            MaxPool2dConfigPy(MaxPool2dConfig::new(kernel_size))
        }

        pub fn with_strides(&self, py: Python<'_>, stride: [usize; 2]) -> MaxPool2dConfigPy {
            MaxPool2dConfigPy(self.0.clone().with_strides(stride))
        }

        pub fn with_padding(
            &mut self,
            py: Python<'_>,
            padding: PaddingConfig2dPy,
        ) -> MaxPool2dConfigPy {
            match padding.0 {
                PaddingConfig2d::Same => {
                    MaxPool2dConfigPy(self.0.clone().with_padding(PaddingConfig2d::Same))
                }
                PaddingConfig2d::Valid => {
                    MaxPool2dConfigPy(self.0.clone().with_padding(PaddingConfig2d::Valid))
                }
                PaddingConfig2d::Explicit(val1, val2) => MaxPool2dConfigPy(
                    self.0
                        .clone()
                        .with_padding(PaddingConfig2d::Explicit(val1, val2)),
                ),
            }
        }

        pub fn with_dilation(&self, dilation: [usize; 2]) -> MaxPool2dConfigPy {
            MaxPool2dConfigPy(self.0.clone().with_dilation(dilation))
        }

        fn init(&self) -> MaxPool2dPy {
            MaxPool2dPy(self.0.init())
        }
    }
}

pub mod interpolate_exports {
    use super::*;
    use burn::nn::interpolate::*;

    for_normal_struct_enums!(
        Interpolate1dPy,
        Interpolate1d,
        "
Interpolate module for resizing 1D tensors with shape [N, C, L]"
    );

    #[pymethods]
    impl Interpolate1dPy {
        #[new]
        #[pyo3(signature = (output_size = None, scale_factor = None, mode = InterpolateModePy::nearest()))]
        fn new(
            output_size: Option<usize>,
            scale_factor: Option<f32>,
            mode: Option<InterpolateModePy>,
        ) -> Self {
            let mode = mode.unwrap_or(InterpolateMode::new_nearest().into());
            Interpolate1dConfig::new()
                .with_output_size(output_size)
                .with_scale_factor(scale_factor)
                .with_mode(mode.0)
                .init()
                .into()
        }

        fn forward(&self, input: TensorPy) -> PyResult<TensorPy> {
            match input {
                TensorPy::TensorThree(tensor) => Ok(self.0.forward(tensor.inner).into()),
                _ => Err(TensorError::NonApplicableMethod.into()),
            }
        }
    }

    for_normal_struct_enums!(
        Interpolate2dPy,
        Interpolate2d,
        "
Interpolate module for resizing tensors with shape [N, C, H, W]."
    );

    #[pymethods]
    impl Interpolate2dPy {
        #[new]
        #[pyo3(signature = (output_size = None, scale_factor = None, mode = InterpolateModePy::nearest()))]
        fn new(
            output_size: Option<[usize; 2]>,
            scale_factor: Option<[f32; 2]>,
            mode: Option<InterpolateModePy>,
        ) -> Self {
            let mode = mode.unwrap_or(InterpolateMode::new_nearest().into());
            Interpolate2dConfig::new()
                .with_output_size(output_size)
                .with_scale_factor(scale_factor)
                .with_mode(mode.0)
                .init()
                .into()
        }

        fn forward(&self, input: TensorPy) -> PyResult<TensorPy> {
            match input {
                TensorPy::TensorFour(tensor) => Ok(self.0.forward(tensor.inner).into()),
                _ => Err(TensorError::NonApplicableMethod.into()),
            }
        }
    }

    for_normal_struct_enums!(
        InterpolateModePy,
        InterpolateMode,
        "
Algorithm used for downsampling and upsampling"
    );

    #[pymethods]
    impl InterpolateModePy {
        #[staticmethod]
        fn nearest() -> Self {
            InterpolateMode::new_nearest().into()
        }

        #[staticmethod]
        fn cubic() -> Self {
            InterpolateMode::new_cubic().into()
        }

        #[staticmethod]
        fn linear() -> Self {
            InterpolateMode::new_linear().into()
        }
    }
}
