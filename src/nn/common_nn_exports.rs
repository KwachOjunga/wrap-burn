use crate::{for_normal_struct_enums};
use super::wgpu_nn_exports::PyTensor;
use super::tensor_error::TensorError;
use burn::nn::*;
use burn::prelude::*;
use pyo3::prelude::*;
use pyo3::types::PyInt;

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


    //[NOTE**] PyAdaptivePool2d

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
