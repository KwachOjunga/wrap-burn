use super::wgpu_nn_exports::*;
use crate::for_normal_struct_enums;
use crate::tensor::base::*;
use crate::tensor::tensor_error::TensorError;
use burn::nn::*;
use burn::prelude::*;
use pyo3::prelude::*;
use pyo3::pyclass_init::PyClassInitializer;
use pyo3::types::PyInt;

// pub fn into_inner<T,U>(wrapper: T) -> U {
//     wrapper.0
// }

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

    impl From<AdaptiveAvgPool1d> for AdaptiveAvgPool1dPy {
        fn from(other: AdaptiveAvgPool1d) -> Self {
            Self(other)
        }
    }

    #[pymethods]
    impl AdaptiveAvgPool1dPy {
        #[getter]
        fn output(&self) -> PyResult<usize> {
            Ok(self.0.output_size)
        }

        #[new]
        fn new(output: usize) -> Self {
            AdaptiveAvgPool1dConfigPy::new(output)
        }

        /// Perform a feedforward tensor operation on a 3 dimensional tensor
        fn forward(&self, tensor: TensorPy) -> PyResult<TensorPy> {
            match tensor {
                TensorPy::TensorThree(val) => Ok(self.0.forward(val.inner).into()),
                _ => Err(TensorError::WrongDimensions.into()),
            }
        }
    }

    #[pymethods]
    impl AdaptiveAvgPool1dConfigPy {
        /// create a new AdaptiveAvgPool1d layer with the given output size
        #[staticmethod]
        fn new(output: usize) -> AdaptiveAvgPool1dPy {
            let mut pool_layer = AdaptiveAvgPool1dConfig::new(output);
            pool_layer.init().into()
        }
    }

    //[NOTE**] PyAdaptiveAvgPool2d

    impl From<AdaptiveAvgPool2d> for AdaptiveAvgPool2dPy {
        fn from(other: AdaptiveAvgPool2d) -> Self {
            Self(other)
        }
    }

    #[pymethods]
    impl AdaptiveAvgPool2dPy {
        #[getter]
        fn output(&self) -> PyResult<[usize; 2]> {
            Ok(self.0.output_size)
        }

        #[new]
        fn new(output: [usize; 2]) -> Self {
            AdaptiveAvgPool2dConfigPy::new(output)
        }

        /// Perform a feedforward tensor operation on a 3 dimensional tensor
        fn forward(&self, tensor: TensorPy) -> PyResult<TensorPy> {
            match tensor {
                TensorPy::TensorFour(val) => Ok(self.0.forward(val.inner).into()),
                _ => Err(TensorError::WrongDimensions.into()),
            }
        }
    }

    #[pymethods]
    impl AdaptiveAvgPool2dConfigPy {
        /// create a new AdaptiveAvgPool1d layer with the given output size
        #[staticmethod]
        fn new(output: [usize; 2]) -> AdaptiveAvgPool2dPy {
            let mut pool_layer = AdaptiveAvgPool2dConfig::new(output);
            pool_layer.init().into()
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
        #[staticmethod]
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
        #[staticmethod]
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
    for_normal_struct_enums!(
        Interpolate1dConfigPy,
        Interpolate1dConfig,
        "Configuration for the 1D interpolation module."
    );
    for_normal_struct_enums!(
        Interpolate2dPy,
        Interpolate2d,
        "
Interpolate module for resizing tensors with shape [N, C, H, W]."
    );
    for_normal_struct_enums!(
        Interpolate2dConfigPy,
        Interpolate2dConfig,
        "
Configuration for the 2D interpolation "
    );
    for_normal_struct_enums!(
        InterpolateModePy,
        InterpolateMode,
        "
Algorithm used for downsampling and upsampling"
    );
}
