//! Warning. The current implementation of TensorPy is grossly inefficient.

use std::f32;

// use std::sync::{Arc, Mutex};
use super::tensor_error::*;
use crate:: impl_tensor_conversions_wgpu;
use burn::backend::Wgpu;
use burn::prelude::*;
use pyo3::prelude::*;

#[derive(Clone, Debug)]
#[pyclass]
pub struct Tensor1 {
    pub inner: Tensor<Wgpu, 1>,
}

#[derive(Clone, Debug)]
#[pyclass]
pub struct Tensor1Bool {
    pub inner: Tensor<Wgpu, 1, Bool>,
}

#[derive(Clone)]
#[pyclass]
pub struct Tensor2 {
    pub inner: Tensor<Wgpu, 2>,
}

#[derive(Clone)]
#[pyclass]
pub struct Tensor2Bool {
    pub inner: Tensor<Wgpu, 2, Bool>,
}

#[derive(Clone)]
#[pyclass]
pub struct Tensor3 {
    pub inner: Tensor<Wgpu, 3>,
}

#[derive(Clone)]
#[pyclass]
pub struct Tensor3Bool {
    pub inner: Tensor<Wgpu, 3, Bool>,
}

#[derive(Clone)]
#[pyclass]
pub struct Tensor4 {
    pub inner: Tensor<Wgpu, 4>,
}

#[derive(Clone)]
#[pyclass]
pub struct Tensor4Bool {
    pub inner: Tensor<Wgpu, 4, Bool>,
}

#[derive(Clone)]
#[pyclass]
pub struct Tensor5 {
    pub inner: Tensor<Wgpu, 5>,
}

#[derive(Clone)]
#[pyclass]
pub struct Tensor5Bool {
    pub inner: Tensor<Wgpu, 5, Bool>,
}

/// A non-idiomatic struct

#[pyclass]
#[non_exhaustive]
#[derive(Clone)]
pub enum TensorPy {
    TensorOne(Tensor1),
    TensorOneBool(Tensor1Bool),
    TensorTwo(Tensor2),
    TensorTwoBool(Tensor2Bool),
    TensorThree(Tensor3),
    TensorThreeBool(Tensor3Bool),
    TensorFour(Tensor4),
    TensorFourBool(Tensor4Bool),
    TensorFive(Tensor5),
    TensorFiveBool(Tensor5Bool),
}

#[pymethods]
impl TensorPy {
    /// Yields an absolute value on a Tensor.
    ///
    /// [note] Non-existent on boolean tensors
    fn abs(&self) -> Option<Self> {
        match self {
            TensorPy::TensorOne(val) => Some(Into::<TensorPy>::into(val.inner.clone().abs())),
            TensorPy::TensorTwo(val) => Some(Into::<TensorPy>::into(val.inner.clone().abs())),
            TensorPy::TensorThree(val) => Some(Into::<TensorPy>::into(val.inner.clone().abs())),
            TensorPy::TensorFour(val) => Some(Into::<TensorPy>::into(val.inner.clone().abs())),
            TensorPy::TensorFive(val) => Some(Into::<TensorPy>::into(val.inner.clone().abs())),
            _ => None,
        }
    }

    /// Non-existent on Boolean tensors
    /// Performs addition on tensors of similar dimensions
    fn add(&self, other: TensorPy) -> Option<Self> {
        match self {
            TensorPy::TensorOne(val) => Some(Into::<TensorPy>::into(
                val.inner.clone().add(
                    Into::<anyhow::Result<Tensor<Wgpu, 1>>>::into(other)
                        .expect("expected 1 dim tensor"),
                ),
            )),
            TensorPy::TensorTwo(val) => Some(Into::<TensorPy>::into(
                val.inner.clone().add(
                    Into::<anyhow::Result<Tensor<Wgpu, 2>>>::into(other)
                        .expect("expected 2 dim tensor"),
                ),
            )),
            TensorPy::TensorThree(val) => Some(Into::<TensorPy>::into(
                val.inner.clone().add(
                    Into::<anyhow::Result<Tensor<Wgpu, 3>>>::into(other)
                        .expect("expected 3 dim tensor"),
                ),
            )),
            TensorPy::TensorFour(val) => Some(Into::<TensorPy>::into(
                val.inner.clone().add(
                    Into::<anyhow::Result<Tensor<Wgpu, 4>>>::into(other)
                        .expect("expected 4 dim tensor"),
                ),
            )),
            TensorPy::TensorFive(val) => Some(Into::<TensorPy>::into(
                val.inner.clone().add(
                    Into::<anyhow::Result<Tensor<Wgpu, 5>>>::into(other)
                        .expect("expected 5 dim tensor"),
                ),
            )),
            _ => None,
        }
    }

    /// Non-existent in tensors whose type is Boolean.
    /// It performs element-wise addition on a tensor.
    fn add_scalar(&self, input: f32) -> Option<Self> {
        match self {
            TensorPy::TensorOne(val) => {
                Some(Into::<TensorPy>::into(val.inner.clone().add_scalar(input)))
            }
            TensorPy::TensorTwo(val) => {
                Some(Into::<TensorPy>::into(val.inner.clone().add_scalar(input)))
            }
            TensorPy::TensorThree(val) => {
                Some(Into::<TensorPy>::into(val.inner.clone().add_scalar(input)))
            }
            TensorPy::TensorFour(val) => {
                Some(Into::<TensorPy>::into(val.inner.clone().add_scalar(input)))
            }
            TensorPy::TensorFive(val) => {
                Some(Into::<TensorPy>::into(val.inner.clone().add_scalar(input)))
            }
            _ => None,
        }
    }

    /// Performs subtraction between a tensors of similar dimensions
    fn sub(&self, other: TensorPy) -> Option<Self> {
        match self {
            TensorPy::TensorOne(val) => Some(Into::<TensorPy>::into(
                val.inner.clone().sub(
                    Into::<anyhow::Result<Tensor<Wgpu, 1>>>::into(other)
                        .expect("expected 1 dim tensor"),
                ),
            )),
            TensorPy::TensorTwo(val) => Some(Into::<TensorPy>::into(
                val.inner.clone().sub(
                    Into::<anyhow::Result<Tensor<Wgpu, 2>>>::into(other)
                        .expect("expected 2 dim tensor"),
                ),
            )),
            TensorPy::TensorThree(val) => Some(Into::<TensorPy>::into(
                val.inner.clone().sub(
                    Into::<anyhow::Result<Tensor<Wgpu, 3>>>::into(other)
                        .expect("expected 3 dim tensor"),
                ),
            )),
            TensorPy::TensorFour(val) => Some(Into::<TensorPy>::into(
                val.inner.clone().sub(
                    Into::<anyhow::Result<Tensor<Wgpu, 4>>>::into(other)
                        .expect("expected 4 dim tensor"),
                ),
            )),
            TensorPy::TensorFive(val) => Some(Into::<TensorPy>::into(
                val.inner.clone().sub(
                    Into::<anyhow::Result<Tensor<Wgpu, 5>>>::into(other)
                        .expect("expected 5 dim tensor"),
                ),
            )),
            _ => None,
        }
    }

    fn sub_scalar(&self, input: f32) -> Option<Self> {
        match self {
            TensorPy::TensorOne(val) => {
                Some(Into::<TensorPy>::into(val.inner.clone().sub_scalar(input)))
            }
            TensorPy::TensorTwo(val) => {
                Some(Into::<TensorPy>::into(val.inner.clone().sub_scalar(input)))
            }
            TensorPy::TensorThree(val) => {
                Some(Into::<TensorPy>::into(val.inner.clone().sub_scalar(input)))
            }
            TensorPy::TensorFour(val) => {
                Some(Into::<TensorPy>::into(val.inner.clone().sub_scalar(input)))
            }
            TensorPy::TensorFive(val) => {
                Some(Into::<TensorPy>::into(val.inner.clone().sub_scalar(input)))
            }
            _ => None,
        }
    }

    fn all_dim(&self, dim: usize) -> Self {
        match self {
            TensorPy::TensorOne(val) => Into::<TensorPy>::into(val.inner.clone().all_dim(dim)),
            TensorPy::TensorTwoBool(val) => Into::<TensorPy>::into(val.inner.clone().all_dim(dim)),
            TensorPy::TensorThreeBool(val) => {
                Into::<TensorPy>::into(val.inner.clone().all_dim(dim))
            }
            TensorPy::TensorFourBool(val) => Into::<TensorPy>::into(val.inner.clone().all_dim(dim)),
            TensorPy::TensorFiveBool(val) => Into::<TensorPy>::into(val.inner.clone().all_dim(dim)),
            TensorPy::TensorOneBool(val) => Into::<TensorPy>::into(val.inner.clone().all_dim(dim)),
            TensorPy::TensorTwo(val) => Into::<TensorPy>::into(val.inner.clone().all_dim(dim)),
            TensorPy::TensorThree(val) => Into::<TensorPy>::into(val.inner.clone().all_dim(dim)),
            TensorPy::TensorFour(val) => Into::<TensorPy>::into(val.inner.clone().all_dim(dim)),
            TensorPy::TensorFive(val) => Into::<TensorPy>::into(val.inner.clone().all_dim(dim)),
        }
    }

    /// Test if any element in the Tensor evaluates to True
    fn any(&self) -> Option<Self> {
        match self {
            TensorPy::TensorOne(val) => Some(Into::<TensorPy>::into(val.inner.clone().any())),
            TensorPy::TensorTwo(val) => Some(Into::<TensorPy>::into(val.inner.clone().any())),
            TensorPy::TensorThree(val) => Some(Into::<TensorPy>::into(val.inner.clone().any())),
            TensorPy::TensorFour(val) => Some(Into::<TensorPy>::into(val.inner.clone().any())),
            TensorPy::TensorFive(val) => Some(Into::<TensorPy>::into(val.inner.clone().any())),
            _ => None,
        }
    }

    fn all(&self) -> Option<Self> {
        match self {
            TensorPy::TensorOne(val) => Some(Into::<TensorPy>::into(val.inner.clone().all())),
            TensorPy::TensorTwo(val) => Some(Into::<TensorPy>::into(val.inner.clone().all())),
            TensorPy::TensorThree(val) => Some(Into::<TensorPy>::into(val.inner.clone().all())),
            TensorPy::TensorFour(val) => Some(Into::<TensorPy>::into(val.inner.clone().all())),
            TensorPy::TensorFive(val) => Some(Into::<TensorPy>::into(val.inner.clone().all())),
            _ => None,
        }
    }

    fn contains_nan(&self) -> PyResult<Self> {
        match self {
            TensorPy::TensorOne(val) => {
                Ok(Into::<TensorPy>::into(val.inner.clone().contains_nan()))
            }
            TensorPy::TensorTwo(val) => {
                Ok(Into::<TensorPy>::into(val.inner.clone().contains_nan()))
            }
            TensorPy::TensorThree(val) => {
                Ok(Into::<TensorPy>::into(val.inner.clone().contains_nan()))
            }
            TensorPy::TensorFour(val) => {
                Ok(Into::<TensorPy>::into(val.inner.clone().contains_nan()))
            }
            TensorPy::TensorFive(val) => {
                Ok(Into::<TensorPy>::into(val.inner.clone().contains_nan()))
            }
            _ => Err(TensorError::NonApplicableMethod.into()),
        }
    }

    fn is_nan(&self) -> PyResult<Self> {
        match self {
            TensorPy::TensorOne(val) => Ok(Into::<TensorPy>::into(val.inner.clone().is_nan())),
            TensorPy::TensorTwo(val) => Ok(Into::<TensorPy>::into(val.inner.clone().is_nan())),
            TensorPy::TensorThree(val) => Ok(Into::<TensorPy>::into(val.inner.clone().is_nan())),
            TensorPy::TensorFour(val) => Ok(Into::<TensorPy>::into(val.inner.clone().is_nan())),
            TensorPy::TensorFive(val) => Ok(Into::<TensorPy>::into(val.inner.clone().is_nan())),
            _ => Err(TensorError::NonApplicableMethod.into()),
        }
    }

    fn is_inf(&self) -> PyResult<Self> {
        match self {
            TensorPy::TensorOne(val) => Ok(Into::<TensorPy>::into(val.inner.clone().is_inf())),
            TensorPy::TensorTwo(val) => Ok(Into::<TensorPy>::into(val.inner.clone().is_inf())),
            TensorPy::TensorThree(val) => Ok(Into::<TensorPy>::into(val.inner.clone().is_inf())),
            TensorPy::TensorFour(val) => Ok(Into::<TensorPy>::into(val.inner.clone().is_inf())),
            TensorPy::TensorFive(val) => Ok(Into::<TensorPy>::into(val.inner.clone().is_inf())),
            _ => Err(TensorError::NonApplicableMethod.into()),
        }
    }

    fn is_finite(&self) -> PyResult<Self> {
        match self {
            TensorPy::TensorOne(val) => Ok(Into::<TensorPy>::into(val.inner.clone().is_finite())),
            TensorPy::TensorTwo(val) => Ok(Into::<TensorPy>::into(val.inner.clone().is_finite())),
            TensorPy::TensorThree(val) => Ok(Into::<TensorPy>::into(val.inner.clone().is_finite())),
            TensorPy::TensorFour(val) => Ok(Into::<TensorPy>::into(val.inner.clone().is_finite())),
            TensorPy::TensorFive(val) => Ok(Into::<TensorPy>::into(val.inner.clone().is_finite())),
            _ => Err(TensorError::NonApplicableMethod.into()),
        }
    }
}

// Use the macro for each dimension
impl_tensor_conversions_wgpu!(Tensor1, Tensor1Bool, 1, TensorOne, TensorOneBool);
impl_tensor_conversions_wgpu!(Tensor2, Tensor2Bool, 2, TensorTwo, TensorTwoBool);
impl_tensor_conversions_wgpu!(Tensor3, Tensor3Bool, 3, TensorThree, TensorThreeBool);
impl_tensor_conversions_wgpu!(Tensor4, Tensor4Bool, 4, TensorFour, TensorFourBool);
impl_tensor_conversions_wgpu!(Tensor5, Tensor5Bool, 5, TensorFive, TensorFiveBool);

mod quantization_exports {
    use crate::{for_normal_struct_enums, implement_wgpu_interface};
    use burn::tensor::quantization::*;

    use super::*;

    implement_wgpu_interface!(
        CalibrationRangePy,
        CalibrationRange,
        "The observed input calibration range"
    );
    implement_wgpu_interface!(
        QuntizationParametesrPrimitivePy,
        QuantizationParametersPrimitive,
        "The quantization parameter primitive"
    );
    // for_normal_struct_enums!(
    //     QuantizedBytesPy,
    //     QuantizedBytes,
    //     "Quantized data bytes representation."
    // );

    // for_normal_struct_enums!(
    //     CalibrationPy,
    //     Calibration,
    //     "The calibration method used to compute the quantization mapping.\nThe method currently in use is the MinMax"
    // );

    // Currently built to run under the specified strategy.
    // for_normal_struct_enums!(QuantizationPy, Quantization, "The quantization method used to quantize the data.\nThe method currently in use is the Symmetric");
    for_normal_struct_enums!(
        QuantAccPrecisionPy,
        QuantAccPrecision,
        "This is the precison used when accumulating values while executing algorithms such as matmul."
    );
    for_normal_struct_enums!(QuantInputTypePy, QuantInputType, "Data type used to represent quantized values");
    for_normal_struct_enums!(QuantLevelPy, QuantLevel, "Level of granularity of quantization");
    for_normal_struct_enums!(QuandtModePy, QuantMode, "Strategy used to quantize values.");


}

#[cfg(test)]
mod tensor_base_tests {
    use super::*;

    #[test]
    fn size_of_tensor() {
        println!("TensorPy size is {}", std::mem::size_of::<TensorPy>());
        println!("Tensor1 size is {}", std::mem::size_of::<Tensor1>());
        println!("Tensor1Bool size is {}", std::mem::size_of::<Tensor1Bool>());
        println!("Tensor2 size is {}", std::mem::size_of::<Tensor2>());
        println!("Tensor2Bool size is {}", std::mem::size_of::<Tensor2Bool>());
        println!("Tensor3 size is {}", std::mem::size_of::<Tensor3>());
        println!("Tensor3Bool size is {}", std::mem::size_of::<Tensor3Bool>());
        println!("Tensor4 size is {}", std::mem::size_of::<Tensor4>());
        println!("Tensor4Bool size is {}", std::mem::size_of::<Tensor4Bool>());
        println!("Tensor5 size is {}", std::mem::size_of::<Tensor5>());
        println!("Tensor5Bool size is {}", std::mem::size_of::<Tensor5Bool>());
    }
}
