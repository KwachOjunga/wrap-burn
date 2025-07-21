use crate::{for_normal_struct_enums};
use super::wgpu_nn_exports::*;
use super::tensor_error::TensorError;
use burn::nn::*;
use burn::prelude::*;
use pyo3::prelude::*;
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
        PyAdaptiveAvgPool1d,
        AdaptiveAvgPool1d,
        "Applies a 1D adaptive avg pooling over input tensors"
    );
   
    for_normal_struct_enums!(
        PyAdaptiveAvgPool1dConfig,
        AdaptiveAvgPool1dConfig,
        "Configuration to create a 1D adaptive avg pooling layer"
    );

    for_normal_struct_enums!(
        PyAdaptiveAvgPool2d,
        AdaptiveAvgPool2d,
        "Applies a 2D adaptive avg pooling over input tensors"
    );
    for_normal_struct_enums!(
        PyAdaptiveAvgPool2dConfig,
        AdaptiveAvgPool2dConfig,
        "Configuration to create a 2D adaptive avg pooling layer"
    );
    for_normal_struct_enums!(
        PyAvgPool1d,
        AvgPool1d,
        "Applies a 1D avg pooling over input tensors."
    );
    for_normal_struct_enums!(
        PyAvgPool1dConfig,
        AvgPool1dConfig,
        "
Configuration to create a 1D avg pooling layer"
    );
    for_normal_struct_enums!(
        PyAvgPool2d,
        AvgPool2d,
        "Applies a 2D avg pooling over input tensors."
    );
    for_normal_struct_enums!(
        PyAvgPool2dConfig,
        AvgPool2dConfig,
        "Configuration to create a 2D avg pooling layer"
    );
    for_normal_struct_enums!(
        PyMaxPool1d,
        MaxPool1d,
        "Applies a 1D max pooling over input tensors."
    );
    for_normal_struct_enums!(
        PyMaxPool1dConfig,
        MaxPool1dConfig,
        "
Configuration to create a 1D max pooling layer"
    );
    for_normal_struct_enums!(
        PyMaxPool2d,
        MaxPool2d,
        "
Applies a 2D max pooling over input tensors."
    );
    for_normal_struct_enums!(
        PyMaxPool2dConfig,
        MaxPool2dConfig,
        "Configuration to create a 2D max pooling layer "
    );


    // Methods section
    // PyAdaptivePool1d

    impl From<AdaptiveAvgPool1d> for PyAdaptiveAvgPool1d {
        fn from(other: AdaptiveAvgPool1d) -> Self {
            Self(other)
        }
    }

    #[pymethods]
    impl PyAdaptiveAvgPool1d {
        #[getter]
        fn output(&self) -> PyResult<usize> {
            Ok(self.0.output_size)
        }

        #[staticmethod]
        fn new(output: usize) -> Self {
            PyAdaptiveAvgPool1dConfig::new(output)
        }

        /// Perform a feedforward tensor operation on a 3 dimensional tensor
        fn forward(&self, tensor: PyTensor) -> PyResult<PyTensor> {
            match tensor {
                PyTensor::TensorThree(val) => {Ok(self.0.forward(val.inner).into())},
                _ => Err(TensorError::WrongDimensions.into())
            }
        }
    }

    #[pymethods]
    impl PyAdaptiveAvgPool1dConfig {
        /// create a new AdaptiveAvgPool1d layer with the given output size
        #[staticmethod]
        fn new(output: usize) -> PyAdaptiveAvgPool1d {
            let mut pool_layer = AdaptiveAvgPool1dConfig::new(output);
            pool_layer.init().into()
        }
    }


    //[NOTE**] PyAdaptiveAvgPool2d

    impl From<AdaptiveAvgPool2d> for PyAdaptiveAvgPool2d {
        fn from(other: AdaptiveAvgPool2d) -> Self {
            Self(other)
        }
    }

    #[pymethods]
    impl PyAdaptiveAvgPool2d {
        #[getter]
        fn output(&self) -> PyResult<[usize; 2]> {
            Ok(self.0.output_size)
        }

        #[staticmethod]
        fn new(output: [usize; 2]) -> Self {
            PyAdaptiveAvgPool2dConfig::new(output)
        }

        /// Perform a feedforward tensor operation on a 3 dimensional tensor
        fn forward(&self, tensor: PyTensor) -> PyResult<PyTensor> {
            match tensor {
                PyTensor::TensorFour(val) => {Ok(self.0.forward(val.inner).into())},
                _ => Err(TensorError::WrongDimensions.into())
            }
        }
    }

    #[pymethods]
    impl PyAdaptiveAvgPool2dConfig {
        /// create a new AdaptiveAvgPool1d layer with the given output size
        #[staticmethod]
        fn new(output: [usize; 2]) -> PyAdaptiveAvgPool2d {
            let mut pool_layer = AdaptiveAvgPool2dConfig::new(output);
            pool_layer.init().into()
        }
    }

    // [NOTE**] PyAvgPool1d
    #[pymethods]
    impl PyAvgPool1d {
        // #[classmethod]
        #[staticmethod]
        #[pyo3(signature = (kernel_size , stride = None, padding = None, count_bool_pad = None))]
        fn new(py: Python<'_>, kernel_size: usize, stride: Option<usize>, padding: Option<PyPaddingConfig1d>, count_bool_pad: Option<bool> ) -> PyAvgPool1d {
            let stride = stride.unwrap_or(1);
            let padding = padding.unwrap_or(PyPaddingConfig1d::valid());
            let count_bool_pad = count_bool_pad.unwrap_or(true);

            PyAvgPool1dConfig::new(kernel_size).with_stride(py, stride).with_padding(py, padding).with_count_include_pad(count_bool_pad).init()
        }
    }

    #[pymethods]
    impl PyAvgPool1dConfig {
        #[staticmethod]
        pub fn new(kernel_size: usize) -> PyAvgPool1dConfig {
            PyAvgPool1dConfig(AvgPool1dConfig::new(kernel_size))
        }

        pub fn with_stride(&self, py: Python<'_>, stride: usize) -> PyAvgPool1dConfig {
            PyAvgPool1dConfig(self.0.clone().with_stride(stride))
        }

        pub fn with_padding(&mut self, py: Python<'_>, padding: PyPaddingConfig1d) -> PyAvgPool1dConfig {
            match padding.0 {
                PaddingConfig1d::Same => PyAvgPool1dConfig(self.0.clone().with_padding(PaddingConfig1d::Same)),
                PaddingConfig1d::Valid => PyAvgPool1dConfig(self.0.clone().with_padding(PaddingConfig1d::Valid)),
                PaddingConfig1d::Explicit(val) => PyAvgPool1dConfig(self.0.clone().with_padding(PaddingConfig1d::Explicit(val)))
            }
        }

        pub fn with_count_include_pad(&self, pad: bool) -> PyAvgPool1dConfig {
            PyAvgPool1dConfig(self.0.clone().with_count_include_pad(pad))
        }

        fn init(&self) -> PyAvgPool1d {
            PyAvgPool1d(self.0.init())
        }
    }


       //[NOTE**] PyAvgPool2d
 
    #[pymethods]
    impl PyAvgPool2d {
        // #[classmethod]
        #[staticmethod]
        #[pyo3(signature = (kernel_size , stride = None, padding = None, count_bool_pad = None))]
        fn new(py: Python<'_>, kernel_size:[usize; 2], stride: Option<[usize; 2]>, padding: Option<PyPaddingConfig2d>, count_bool_pad: Option<bool> ) -> PyAvgPool2d {
            let stride = stride.unwrap_or([1,1]);
            let padding = padding.unwrap_or(PyPaddingConfig2d::valid());
            let count_bool_pad = count_bool_pad.unwrap_or(true);

            PyAvgPool2dConfig::new(kernel_size).with_strides(py, stride).with_padding(py, padding).with_count_include_pad(count_bool_pad).init()
        }
    }


    #[pymethods]
    impl PyAvgPool2dConfig {
        #[staticmethod]
        pub fn new(kernel_size: [usize; 2]) -> PyAvgPool2dConfig {
            PyAvgPool2dConfig(AvgPool2dConfig::new(kernel_size))
        }

        pub fn with_strides(&self, py: Python<'_>, stride: [usize; 2]) -> PyAvgPool2dConfig {
            PyAvgPool2dConfig(self.0.clone().with_strides(stride))
        }

        pub fn with_padding(&mut self, py: Python<'_>, padding: PyPaddingConfig2d) -> PyAvgPool2dConfig {
            match padding.0 {
                PaddingConfig2d::Same => PyAvgPool2dConfig(self.0.clone().with_padding(PaddingConfig2d::Same)),
                PaddingConfig2d::Valid => PyAvgPool2dConfig(self.0.clone().with_padding(PaddingConfig2d::Valid)),
                PaddingConfig2d::Explicit(val1, val2) => PyAvgPool2dConfig(self.0.clone().with_padding(PaddingConfig2d::Explicit(val1,val2)))
            }
        }

        pub fn with_count_include_pad(&self, pad: bool) -> PyAvgPool2dConfig {
            PyAvgPool2dConfig(self.0.clone().with_count_include_pad(pad))
        }

        fn init(&self) -> PyAvgPool2d {
            PyAvgPool2d(self.0.init())
        }
    }

    //[NOTE**] PyMaxPool1d

    #[pymethods]
    impl PyMaxPool1d {
        // #[classmethod]
        #[staticmethod]
        #[pyo3(signature = (kernel_size , stride = None, padding = None, dilation = Some(1)))]
        fn new(py: Python<'_>, kernel_size: usize, stride: Option<usize>, padding: Option<PyPaddingConfig1d>, dilation: Option<usize> ) -> PyMaxPool1d {
            let stride = stride.unwrap_or(1);
            let padding = padding.unwrap_or(PyPaddingConfig1d::valid());
            let dilation = dilation.unwrap_or(1);

            PyMaxPool1dConfig::new(kernel_size).with_stride(py, stride).with_padding(py, padding).with_dilation(dilation).init()
        }
    }

    #[pymethods]
    impl PyMaxPool1dConfig {
        #[staticmethod]
        pub fn new(kernel_size: usize) -> PyMaxPool1dConfig {
            PyMaxPool1dConfig(MaxPool1dConfig::new(kernel_size))
        }

        pub fn with_stride(&self, py: Python<'_>, stride: usize) -> PyMaxPool1dConfig {
            PyMaxPool1dConfig(self.0.clone().with_stride(stride))
        }

        pub fn with_padding(&mut self, py: Python<'_>, padding: PyPaddingConfig1d) -> PyMaxPool1dConfig {
            match padding.0 {
                PaddingConfig1d::Same => PyMaxPool1dConfig(self.0.clone().with_padding(PaddingConfig1d::Same)),
                PaddingConfig1d::Valid => PyMaxPool1dConfig(self.0.clone().with_padding(PaddingConfig1d::Valid)),
                PaddingConfig1d::Explicit(val) => PyMaxPool1dConfig(self.0.clone().with_padding(PaddingConfig1d::Explicit(val)))
            }
        }

        pub fn with_dilation(&self, dilation: usize) -> PyMaxPool1dConfig {
            PyMaxPool1dConfig(self.0.clone().with_dilation(dilation))
        }

        fn init(&self) -> PyMaxPool1d {
            PyMaxPool1d(self.0.init())
        }
    }


    // [NOTE**] PyMaxPool2d

    #[pymethods]
    impl PyMaxPool2d {
        // #[classmethod]
        #[staticmethod]
        #[pyo3(signature = (kernel_size , stride = None, padding = None, dilation = None))]
        fn new(py: Python<'_>, kernel_size:[usize; 2], stride: Option<[usize; 2]>, padding: Option<PyPaddingConfig2d>, dilation: Option<[usize;2]> ) -> PyMaxPool2d {
            let stride = stride.unwrap_or([1,1]);
            let padding = padding.unwrap_or(PyPaddingConfig2d::valid());
            let dilation = dilation.unwrap_or([1,1]);

            PyMaxPool2dConfig::new(kernel_size).with_strides(py, stride).with_padding(py, padding).with_dilation(dilation).init()
        }
    }


    #[pymethods]
    impl PyMaxPool2dConfig {
        #[staticmethod]
        pub fn new(kernel_size: [usize; 2]) -> PyMaxPool2dConfig {
            PyMaxPool2dConfig(MaxPool2dConfig::new(kernel_size))
        }

        pub fn with_strides(&self, py: Python<'_>, stride: [usize; 2]) -> PyMaxPool2dConfig {
            PyMaxPool2dConfig(self.0.clone().with_strides(stride))
        }

        pub fn with_padding(&mut self, py: Python<'_>, padding: PyPaddingConfig2d) -> PyMaxPool2dConfig {
            match padding.0 {
                PaddingConfig2d::Same => PyMaxPool2dConfig(self.0.clone().with_padding(PaddingConfig2d::Same)),
                PaddingConfig2d::Valid => PyMaxPool2dConfig(self.0.clone().with_padding(PaddingConfig2d::Valid)),
                PaddingConfig2d::Explicit(val1, val2) => PyMaxPool2dConfig(self.0.clone().with_padding(PaddingConfig2d::Explicit(val1,val2)))
            }
        }

        pub fn with_dilation(&self, dilation: [usize;2]) -> PyMaxPool2dConfig {
            PyMaxPool2dConfig(self.0.clone().with_dilation(dilation))
        }

        fn init(&self) -> PyMaxPool2d {
            PyMaxPool2d(self.0.init())
        }
    }

}

pub mod interpolate_exports {
    use super::*;
    use burn::nn::interpolate::*;

    for_normal_struct_enums!(
        PyInterpolate1d,
        Interpolate1d,
        "
Interpolate module for resizing 1D tensors with shape [N, C, L]"
    );
    for_normal_struct_enums!(
        PyInterpolate1dConfig,
        Interpolate1dConfig,
        "Configuration for the 1D interpolation module."
    );
    for_normal_struct_enums!(
        PyInterpolate2d,
        Interpolate2d,
        "
Interpolate module for resizing tensors with shape [N, C, H, W]."
    );
    for_normal_struct_enums!(
        PyInterpolate2dConfig,
        Interpolate2dConfig,
        "
Configuration for the 2D interpolation "
    );
    for_normal_struct_enums!(
        PyInterpolateMode,
        InterpolateMode,
        "
Algorithm used for downsampling and upsampling"
    );

}
