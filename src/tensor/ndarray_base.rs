//! Warning. The current implementation of TensorPy is grossly inefficient.

use std::f32;

use super::common_tensor_exports;
use crate::impl_tensor_conversions_ndarray;
use crate::nn::NDARRAYDEVICE;

use super::tensor_error::*;
use burn::backend::NdArray;
use burn::prelude::*;
use pyo3::prelude::*;

#[derive(Clone, Debug)]
#[pyclass]
pub struct Tensor1 {
    pub inner: Tensor<NdArray, 1>,
}

#[derive(Clone, Debug)]
#[pyclass]
pub struct Tensor1Bool {
    pub inner: Tensor<NdArray, 1, Bool>,
}

#[derive(Clone, Debug)]
#[pyclass]
pub struct Tensor1Int {
    pub inner: Tensor<NdArray, 1, Int>,
}

#[derive(Clone)]
#[pyclass]
pub struct Tensor2 {
    pub inner: Tensor<NdArray, 2>,
}

#[derive(Clone)]
#[pyclass]
pub struct Tensor2Bool {
    pub inner: Tensor<NdArray, 2, Bool>,
}

#[derive(Clone, Debug)]
#[pyclass]
pub struct Tensor2Int {
    pub inner: Tensor<NdArray, 2, Int>,
}

#[derive(Clone)]
#[pyclass]
pub struct Tensor3 {
    pub inner: Tensor<NdArray, 3>,
}

#[derive(Clone)]
#[pyclass]
pub struct Tensor3Bool {
    pub inner: Tensor<NdArray, 3, Bool>,
}

#[derive(Clone, Debug)]
#[pyclass]
pub struct Tensor3Int {
    pub inner: Tensor<NdArray, 3, Int>,
}
#[derive(Clone)]
#[pyclass]
pub struct Tensor4 {
    pub inner: Tensor<NdArray, 4>,
}

#[derive(Clone)]
#[pyclass]
pub struct Tensor4Bool {
    pub inner: Tensor<NdArray, 4, Bool>,
}

#[derive(Clone, Debug)]
#[pyclass]
pub struct Tensor4Int {
    pub inner: Tensor<NdArray, 4, Int>,
}

#[derive(Clone)]
#[pyclass]
pub struct Tensor5 {
    pub inner: Tensor<NdArray, 5>,
}

#[derive(Clone)]
#[pyclass]
pub struct Tensor5Bool {
    pub inner: Tensor<NdArray, 5, Bool>,
}

#[derive(Clone)]
#[pyclass]
pub struct Tensor5Int {
    pub inner: Tensor<NdArray, 5, Int>,
}

/// A non-idiomatic struct

#[pyclass]
#[non_exhaustive]
#[derive(Clone)]
pub enum TensorPy {
    TensorOne(Tensor1),
    TensorOneBool(Tensor1Bool),
    TensorOneInt(Tensor1Int),
    TensorTwo(Tensor2),
    TensorTwoBool(Tensor2Bool),
    TensorTwoInt(Tensor2Int),
    TensorThree(Tensor3),
    TensorThreeBool(Tensor3Bool),
    TensorThreeInt(Tensor3Int),
    TensorFour(Tensor4),
    TensorFourBool(Tensor4Bool),
    TensorFourInt(Tensor4Int),
    TensorFive(Tensor5),
    TensorFiveBool(Tensor5Bool),
    TensorFiveInt(Tensor5Int),
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
            TensorPy::TensorOneInt(val) => Some(Into::<TensorPy>::into(val.inner.clone().abs())),
            TensorPy::TensorTwoInt(val) => Some(Into::<TensorPy>::into(val.inner.clone().abs())),
            TensorPy::TensorThreeInt(val) => Some(Into::<TensorPy>::into(val.inner.clone().abs())),
            TensorPy::TensorFourInt(val) => Some(Into::<TensorPy>::into(val.inner.clone().abs())),
            TensorPy::TensorFiveInt(val) => Some(Into::<TensorPy>::into(val.inner.clone().abs())),
            _ => None,
        }
    }

    /// Non-existent on Boolean tensors
    /// Performs addition on tensors of similar dimensions
    fn add(&self, other: TensorPy) -> Option<Self> {
        match self {
            TensorPy::TensorOne(val) => Some(Into::<TensorPy>::into(
                val.inner.clone().add(
                    Into::<anyhow::Result<Tensor<NdArray, 1>>>::into(other)
                        .expect("expected 1 dim tensor"),
                ),
            )),
            TensorPy::TensorTwo(val) => Some(Into::<TensorPy>::into(
                val.inner.clone().add(
                    Into::<anyhow::Result<Tensor<NdArray, 2>>>::into(other)
                        .expect("expected 2 dim tensor"),
                ),
            )),
            TensorPy::TensorThree(val) => Some(Into::<TensorPy>::into(
                val.inner.clone().add(
                    Into::<anyhow::Result<Tensor<NdArray, 3>>>::into(other)
                        .expect("expected 3 dim tensor"),
                ),
            )),
            TensorPy::TensorFour(val) => Some(Into::<TensorPy>::into(
                val.inner.clone().add(
                    Into::<anyhow::Result<Tensor<NdArray, 4>>>::into(other)
                        .expect("expected 4 dim tensor"),
                ),
            )),
            TensorPy::TensorFive(val) => Some(Into::<TensorPy>::into(
                val.inner.clone().add(
                    Into::<anyhow::Result<Tensor<NdArray, 5>>>::into(other)
                        .expect("expected 5 dim tensor"),
                ),
            )),
            TensorPy::TensorOneInt(val) => Some(Into::<TensorPy>::into(
                val.inner.clone().add(
                    Into::<anyhow::Result<Tensor<NdArray, 1, Int>>>::into(other)
                        .expect("expected 1 dim tensor"),
                ),
            )),
            TensorPy::TensorTwoInt(val) => Some(Into::<TensorPy>::into(
                val.inner.clone().add(
                    Into::<anyhow::Result<Tensor<NdArray, 2, Int>>>::into(other)
                        .expect("expected 2 dim tensor"),
                ),
            )),
            TensorPy::TensorThreeInt(val) => Some(Into::<TensorPy>::into(
                val.inner.clone().add(
                    Into::<anyhow::Result<Tensor<NdArray, 3, Int>>>::into(other)
                        .expect("expected 3 dim tensor"),
                ),
            )),
            TensorPy::TensorFourInt(val) => Some(Into::<TensorPy>::into(
                val.inner.clone().add(
                    Into::<anyhow::Result<Tensor<NdArray, 4, Int>>>::into(other)
                        .expect("expected 4 dim tensor"),
                ),
            )),
            TensorPy::TensorFiveInt(val) => Some(Into::<TensorPy>::into(
                val.inner.clone().add(
                    Into::<anyhow::Result<Tensor<NdArray, 5, Int>>>::into(other)
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
            TensorPy::TensorOneInt(val) => {
                Some(Into::<TensorPy>::into(val.inner.clone().add_scalar(input)))
            }
            TensorPy::TensorTwoInt(val) => {
                Some(Into::<TensorPy>::into(val.inner.clone().add_scalar(input)))
            }
            TensorPy::TensorThreeInt(val) => {
                Some(Into::<TensorPy>::into(val.inner.clone().add_scalar(input)))
            }
            TensorPy::TensorFourInt(val) => {
                Some(Into::<TensorPy>::into(val.inner.clone().add_scalar(input)))
            }
            TensorPy::TensorFiveInt(val) => {
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
                    Into::<anyhow::Result<Tensor<NdArray, 1>>>::into(other)
                        .expect("expected 1 dim tensor"),
                ),
            )),
            TensorPy::TensorTwo(val) => Some(Into::<TensorPy>::into(
                val.inner.clone().sub(
                    Into::<anyhow::Result<Tensor<NdArray, 2>>>::into(other)
                        .expect("expected 2 dim tensor"),
                ),
            )),
            TensorPy::TensorThree(val) => Some(Into::<TensorPy>::into(
                val.inner.clone().sub(
                    Into::<anyhow::Result<Tensor<NdArray, 3>>>::into(other)
                        .expect("expected 3 dim tensor"),
                ),
            )),
            TensorPy::TensorFour(val) => Some(Into::<TensorPy>::into(
                val.inner.clone().sub(
                    Into::<anyhow::Result<Tensor<NdArray, 4>>>::into(other)
                        .expect("expected 4 dim tensor"),
                ),
            )),
            TensorPy::TensorFive(val) => Some(Into::<TensorPy>::into(
                val.inner.clone().sub(
                    Into::<anyhow::Result<Tensor<NdArray, 5>>>::into(other)
                        .expect("expected 5 dim tensor"),
                ),
            )),
            TensorPy::TensorOneInt(val) => Some(Into::<TensorPy>::into(
                val.inner.clone().sub(
                    Into::<anyhow::Result<Tensor<NdArray, 1, Int>>>::into(other)
                        .expect("expected 1 dim tensor"),
                ),
            )),
            TensorPy::TensorTwoInt(val) => Some(Into::<TensorPy>::into(
                val.inner.clone().sub(
                    Into::<anyhow::Result<Tensor<NdArray, 2, Int>>>::into(other)
                        .expect("expected 2 dim tensor"),
                ),
            )),
            TensorPy::TensorThreeInt(val) => Some(Into::<TensorPy>::into(
                val.inner.clone().sub(
                    Into::<anyhow::Result<Tensor<NdArray, 3, Int>>>::into(other)
                        .expect("expected 3 dim tensor"),
                ),
            )),
            TensorPy::TensorFourInt(val) => Some(Into::<TensorPy>::into(
                val.inner.clone().sub(
                    Into::<anyhow::Result<Tensor<NdArray, 4, Int>>>::into(other)
                        .expect("expected 4 dim tensor"),
                ),
            )),
            TensorPy::TensorFiveInt(val) => Some(Into::<TensorPy>::into(
                val.inner.clone().sub(
                    Into::<anyhow::Result<Tensor<NdArray, 5, Int>>>::into(other)
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
            TensorPy::TensorTwoInt(val) => Into::<TensorPy>::into(val.inner.clone().all_dim(dim)),
            TensorPy::TensorThreeInt(val) => Into::<TensorPy>::into(val.inner.clone().all_dim(dim)),
            TensorPy::TensorFourInt(val) => Into::<TensorPy>::into(val.inner.clone().all_dim(dim)),
            TensorPy::TensorFiveInt(val) => Into::<TensorPy>::into(val.inner.clone().all_dim(dim)),
            TensorPy::TensorOneInt(val) => Into::<TensorPy>::into(val.inner.clone().all_dim(dim)),
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
            // TensorPy::TensorOneInt(val) => {
            //     Ok(Into::<TensorPy>::into(val.inner.clone().contains_nan()))
            // }
            // TensorPy::TensorTwoInt(val) => {
            //     Ok(Into::<TensorPy>::into(val.inner.clone().contains_nan()))
            // }
            // TensorPy::TensorThreeInt(val) => {
            //     Ok(Into::<TensorPy>::into(val.inner.clone().contains_nan()))
            // }
            // TensorPy::TensorFourInt(val) => {
            //     Ok(Into::<TensorPy>::into(val.inner.clone().contains_nan()))
            // }
            // TensorPy::TensorFiveInt(val) => {
            //     Ok(Into::<TensorPy>::into(val.inner.clone().contains_nan()))
            // }
            _ => Err(TensorError::NonApplicableMethod.into()),
        }
    }

    /// Prints the shape of the tensor
    fn dims(&self) {
        let dim = match self {
            TensorPy::TensorOne(val) => val.inner.shape(),
            TensorPy::TensorTwo(val) => val.inner.shape(),
            TensorPy::TensorThree(val) => val.inner.shape(),
            TensorPy::TensorFour(val) => val.inner.shape(),
            TensorPy::TensorFive(val) => val.inner.shape(),
            TensorPy::TensorOneBool(val) => val.inner.shape(),
            TensorPy::TensorTwoBool(val) => val.inner.shape(),
            TensorPy::TensorThreeBool(val) => val.inner.shape(),
            TensorPy::TensorFourBool(val) => val.inner.shape(),
            TensorPy::TensorFiveBool(val) => val.inner.shape(),
            TensorPy::TensorOneInt(val) => val.inner.shape(),
            TensorPy::TensorTwoInt(val) => val.inner.shape(),
            TensorPy::TensorThreeInt(val) => val.inner.shape(),
            TensorPy::TensorFourInt(val) => val.inner.shape(),
            TensorPy::TensorFiveInt(val) => val.inner.shape(),
        };
        println!("{:#?}", dim);
    }

    /// Creates an empty tensor provided the shape and dimensions are consistent
    ///
    /// ```python
    ///
    ///     from pb.wgpu.wgpu_tensor import TensorPy
    ///     # this creates a 3 dim tensor whose shape is as given
    ///     x = TensorPy.empty([2,3,4], 3)
    ///     
    /// ```
    #[staticmethod]
    fn empty(shape: Vec<usize>, dim: usize) -> PyResult<Self> {
        match dim {
            1 => Ok(Tensor::<NdArray, 1>::empty(shape, &NDARRAYDEVICE).into()),
            2 => Ok(Tensor::<NdArray, 2>::empty(shape, &NDARRAYDEVICE).into()),
            3 => Ok(Tensor::<NdArray, 3>::empty(shape, &NDARRAYDEVICE).into()),
            4 => Ok(Tensor::<NdArray, 4>::empty(shape, &NDARRAYDEVICE).into()),
            5 => Ok(Tensor::<NdArray, 5>::empty(shape, &NDARRAYDEVICE).into()),
            _ => Err(
                TensorError::WrongDimensions.into(), /*("Unsupported dimensions")*/
            ),
        }
    }

    // [TODO:] Implement this method for Int types.
    /// Generate a tensor of random values whose type is Float
    #[staticmethod]
    fn random(
        shape: Vec<usize>,
        dim: usize,
        dist: Option<common_tensor_exports::Distribution>,
    ) -> PyResult<Self> {
        let distribution = match dist {
            None => burn::tensor::Distribution::Default,
            Some(common_tensor_exports::Distribution::Bernoulli(val)) => {
                burn::tensor::Distribution::Bernoulli(val)
            }
            Some(common_tensor_exports::Distribution::Normal(val1, val2)) => {
                burn::tensor::Distribution::Normal(val1, val2)
            }
            Some(common_tensor_exports::Distribution::Uniform(val1, val2)) => {
                burn::tensor::Distribution::Uniform(val1, val2)
            }
        };
        match dim {
            1 => Ok(Tensor::<NdArray, 1>::random(shape, distribution, &NDARRAYDEVICE).into()),
            2 => Ok(Tensor::<NdArray, 2>::random(shape, distribution, &NDARRAYDEVICE).into()),
            3 => Ok(Tensor::<NdArray, 3>::random(shape, distribution, &NDARRAYDEVICE).into()),
            4 => Ok(Tensor::<NdArray, 4>::random(shape, distribution, &NDARRAYDEVICE).into()),
            5 => Ok(Tensor::<NdArray, 5>::random(shape, distribution, &NDARRAYDEVICE).into()),
            _ => Err(
                TensorError::WrongDimensions.into(), /*("Unsupported dimensions")*/
            ),
        }
    }

    fn is_nan(&self) -> PyResult<Self> {
        match self {
            TensorPy::TensorOne(val) => Ok(Into::<TensorPy>::into(val.inner.clone().is_nan())),
            TensorPy::TensorTwo(val) => Ok(Into::<TensorPy>::into(val.inner.clone().is_nan())),
            TensorPy::TensorThree(val) => Ok(Into::<TensorPy>::into(val.inner.clone().is_nan())),
            TensorPy::TensorFour(val) => Ok(Into::<TensorPy>::into(val.inner.clone().is_nan())),
            TensorPy::TensorFive(val) => Ok(Into::<TensorPy>::into(val.inner.clone().is_nan())),
            // TensorPy::TensorOneInt(val) => Ok(Into::<TensorPy>::into(val.inner.clone().is_nan())),
            // TensorPy::TensorTwoInt(val) => Ok(Into::<TensorPy>::into(val.inner.clone().is_nan())),
            // TensorPy::TensorThreeInt(val) => Ok(Into::<TensorPy>::into(val.inner.clone().is_nan())),
            // TensorPy::TensorFourInt(val) => Ok(Into::<TensorPy>::into(val.inner.clone().is_nan())),
            // TensorPy::TensorFiveInt(val) => Ok(Into::<TensorPy>::into(val.inner.clone().is_nan())),
            _ => Err(TensorError::NonApplicableMethod.into()),
        }
    }

    fn is_inf(&self) -> PyResult<Self> {
        match self {
            TensorPy::TensorOne(val) => Ok(Into::<TensorPy>::into(val.inner.clone().is_inf())),
            TensorPy::TensorTwo(val) => Ok(Into::<TensorPy>::into(val.inner.clone().is_inf())),
            TensorPy::TensorThree(val) => Ok(Into::<TensorPy>::into(val.inner.clone().is_inf())),
            TensorPy::TensorFour(val) => Ok(Into::<TensorPy>::into(val.inner.clone().is_inf())),
            // TensorPy::TensorFive(val) => Ok(Into::<TensorPy>::into(val.inner.clone().is_inf())),
            // TensorPy::TensorOneInt(val) => Ok(Into::<TensorPy>::into(val.inner.clone().is_inf())),
            // TensorPy::TensorTwoInt(val) => Ok(Into::<TensorPy>::into(val.inner.clone().is_inf())),
            // TensorPy::TensorThreeInt(val) => Ok(Into::<TensorPy>::into(val.inner.clone().is_inf())),
            // TensorPy::TensorFourInt(val) => Ok(Into::<TensorPy>::into(val.inner.clone().is_inf())),
            // TensorPy::TensorFiveInt(val) => Ok(Into::<TensorPy>::into(val.inner.clone().is_inf())),
            _ => Err(TensorError::NonApplicableMethod.into()),
        }
    }

    fn is_finite(&self) -> PyResult<Self> {
        match self {
            TensorPy::TensorOne(val) => Ok(Into::<TensorPy>::into(val.inner.clone().is_finite())),
            TensorPy::TensorTwo(val) => Ok(Into::<TensorPy>::into(val.inner.clone().is_finite())),
            TensorPy::TensorThree(val) => Ok(Into::<TensorPy>::into(val.inner.clone().is_finite())),
            TensorPy::TensorFour(val) => Ok(Into::<TensorPy>::into(val.inner.clone().is_finite())),
            // TensorPy::TensorFive(val) => Ok(Into::<TensorPy>::into(val.inner.clone().is_finite())),
            // TensorPy::TensorOneInt(val) => Ok(Into::<TensorPy>::into(val.inner.clone().is_finite())),
            // TensorPy::TensorTwoInt(val) => Ok(Into::<TensorPy>::into(val.inner.clone().is_finite())),
            // TensorPy::TensorThreeInt(val) => Ok(Into::<TensorPy>::into(val.inner.clone().is_finite())),
            // TensorPy::TensorFourInt(val) => Ok(Into::<TensorPy>::into(val.inner.clone().is_finite())),
            // TensorPy::TensorFiveInt(val) => Ok(Into::<TensorPy>::into(val.inner.clone().is_finite())),
            _ => Err(TensorError::NonApplicableMethod.into()),
        }
    }

    fn mean(&self) -> Option<Self> {
        match self {
            TensorPy::TensorOne(val) => Some(Into::<TensorPy>::into(val.inner.clone().mean())),
            TensorPy::TensorTwo(val) => Some(Into::<TensorPy>::into(val.inner.clone().mean())),
            TensorPy::TensorThree(val) => Some(Into::<TensorPy>::into(val.inner.clone().mean())),
            TensorPy::TensorFour(val) => Some(Into::<TensorPy>::into(val.inner.clone().mean())),
            TensorPy::TensorFive(val) => Some(Into::<TensorPy>::into(val.inner.clone().mean())),
            TensorPy::TensorOneInt(val) => Some(Into::<TensorPy>::into(val.inner.clone().mean())),
            TensorPy::TensorTwoInt(val) => Some(Into::<TensorPy>::into(val.inner.clone().mean())),
            TensorPy::TensorThreeInt(val) => Some(Into::<TensorPy>::into(val.inner.clone().mean())),
            TensorPy::TensorFourInt(val) => Some(Into::<TensorPy>::into(val.inner.clone().mean())),
            TensorPy::TensorFiveInt(val) => Some(Into::<TensorPy>::into(val.inner.clone().mean())),
            _ => None,
        }
    }

    /// Aggregate mean along the given dimension
    fn mean_dim(&self, dim: usize) -> Option<Self> {
        match self {
            TensorPy::TensorOne(val) => {
                Some(Into::<TensorPy>::into(val.inner.clone().mean_dim(dim)))
            }
            TensorPy::TensorTwo(val) => {
                Some(Into::<TensorPy>::into(val.inner.clone().mean_dim(dim)))
            }
            TensorPy::TensorThree(val) => {
                Some(Into::<TensorPy>::into(val.inner.clone().mean_dim(dim)))
            }
            TensorPy::TensorFour(val) => {
                Some(Into::<TensorPy>::into(val.inner.clone().mean_dim(dim)))
            }
            TensorPy::TensorFive(val) => {
                Some(Into::<TensorPy>::into(val.inner.clone().mean_dim(dim)))
            }
            TensorPy::TensorOneInt(val) => {
                Some(Into::<TensorPy>::into(val.inner.clone().mean_dim(dim)))
            }
            TensorPy::TensorTwoInt(val) => {
                Some(Into::<TensorPy>::into(val.inner.clone().mean_dim(dim)))
            }
            TensorPy::TensorThreeInt(val) => {
                Some(Into::<TensorPy>::into(val.inner.clone().mean_dim(dim)))
            }
            TensorPy::TensorFourInt(val) => {
                Some(Into::<TensorPy>::into(val.inner.clone().mean_dim(dim)))
            }
            TensorPy::TensorFiveInt(val) => {
                Some(Into::<TensorPy>::into(val.inner.clone().mean_dim(dim)))
            }
            _ => None,
        }
    }

    /// By default this creates a tensor of ones whose type is a float
    #[staticmethod]
    fn ones(shape: Vec<usize>, dim: usize) -> PyResult<Self> {
        match dim {
            1 => Ok(Tensor::<NdArray, 1>::ones(shape, &NDARRAYDEVICE).into()),
            2 => Ok(Tensor::<NdArray, 2>::ones(shape, &NDARRAYDEVICE).into()),
            3 => Ok(Tensor::<NdArray, 3>::ones(shape, &NDARRAYDEVICE).into()),
            4 => Ok(Tensor::<NdArray, 4>::ones(shape, &NDARRAYDEVICE).into()),
            5 => Ok(Tensor::<NdArray, 5>::ones(shape, &NDARRAYDEVICE).into()),
            _ => Err(
                TensorError::WrongDimensions.into(), /*("Unsupported dimensions")*/
            ),
        }
    }

    fn ones_like(&self) -> PyResult<Self> {
        match self {
            TensorPy::TensorOne(val) => Ok(val.inner.ones_like().into()),
            TensorPy::TensorTwo(val) => Ok(val.inner.ones_like().into()),
            TensorPy::TensorThree(val) => Ok(val.inner.ones_like().into()),
            TensorPy::TensorFour(val) => Ok(val.inner.ones_like().into()),
            TensorPy::TensorFive(val) => Ok(val.inner.ones_like().into()),
            TensorPy::TensorOneInt(val) => Ok(val.inner.ones_like().into()),
            TensorPy::TensorTwoInt(val) => Ok(val.inner.ones_like().into()),
            TensorPy::TensorThreeInt(val) => Ok(val.inner.ones_like().into()),
            TensorPy::TensorFourInt(val) => Ok(val.inner.ones_like().into()),
            TensorPy::TensorFiveInt(val) => Ok(val.inner.ones_like().into()),
            _ => Err(TensorError::NonApplicableMethod.into()),
        }
    }

    fn prod(&self) -> Option<Self> {
        match self {
            TensorPy::TensorOne(val) => Some(Into::<TensorPy>::into(val.inner.clone().prod())),
            TensorPy::TensorTwo(val) => Some(Into::<TensorPy>::into(val.inner.clone().prod())),
            TensorPy::TensorThree(val) => Some(Into::<TensorPy>::into(val.inner.clone().prod())),
            TensorPy::TensorFour(val) => Some(Into::<TensorPy>::into(val.inner.clone().prod())),
            TensorPy::TensorFive(val) => Some(Into::<TensorPy>::into(val.inner.clone().prod())),
            _ => None,
        }
    }

    /// Aggregate product along the given dimension
    fn prod_dim(&self, dim: usize) -> Option<Self> {
        match self {
            TensorPy::TensorOne(val) => {
                Some(Into::<TensorPy>::into(val.inner.clone().prod_dim(dim)))
            }
            TensorPy::TensorTwo(val) => {
                Some(Into::<TensorPy>::into(val.inner.clone().prod_dim(dim)))
            }
            TensorPy::TensorThree(val) => {
                Some(Into::<TensorPy>::into(val.inner.clone().prod_dim(dim)))
            }
            TensorPy::TensorFour(val) => {
                Some(Into::<TensorPy>::into(val.inner.clone().prod_dim(dim)))
            }
            TensorPy::TensorFive(val) => {
                Some(Into::<TensorPy>::into(val.inner.clone().prod_dim(dim)))
            }
            _ => None,
        }
    }

    fn sum(&self) -> Option<Self> {
        match self {
            TensorPy::TensorOne(val) => Some(Into::<TensorPy>::into(val.inner.clone().sum())),
            TensorPy::TensorTwo(val) => Some(Into::<TensorPy>::into(val.inner.clone().sum())),
            TensorPy::TensorThree(val) => Some(Into::<TensorPy>::into(val.inner.clone().sum())),
            TensorPy::TensorFour(val) => Some(Into::<TensorPy>::into(val.inner.clone().sum())),
            TensorPy::TensorFive(val) => Some(Into::<TensorPy>::into(val.inner.clone().sum())),
            _ => None,
        }
    }

    fn sum_dim(&self, dim: usize) -> Option<Self> {
        match self {
            TensorPy::TensorOne(val) => {
                Some(Into::<TensorPy>::into(val.inner.clone().sum_dim(dim)))
            }
            TensorPy::TensorTwo(val) => {
                Some(Into::<TensorPy>::into(val.inner.clone().sum_dim(dim)))
            }
            TensorPy::TensorThree(val) => {
                Some(Into::<TensorPy>::into(val.inner.clone().sum_dim(dim)))
            }
            TensorPy::TensorFour(val) => {
                Some(Into::<TensorPy>::into(val.inner.clone().sum_dim(dim)))
            }
            TensorPy::TensorFive(val) => {
                Some(Into::<TensorPy>::into(val.inner.clone().sum_dim(dim)))
            }
            _ => None,
        }
    }

    /// Creates a tensor with zeros provided the shape and dimensions are consistent
    ///
    /// ```python
    ///
    ///     from pb.wgpu.wgpu_tensor import TensorPy
    ///     # this creates a 3 dim tensor whose shape is as given
    ///     x = TensorPy.zeros([2,3,4], 3)
    ///     
    /// ```
    #[staticmethod]
    fn zeros(shape: Vec<usize>, dim: usize) -> PyResult<Self> {
        match dim {
            1 => Ok(Tensor::<NdArray, 1>::zeros(shape, &NDARRAYDEVICE).into()),
            2 => Ok(Tensor::<NdArray, 2>::zeros(shape, &NDARRAYDEVICE).into()),
            3 => Ok(Tensor::<NdArray, 3>::zeros(shape, &NDARRAYDEVICE).into()),
            4 => Ok(Tensor::<NdArray, 4>::zeros(shape, &NDARRAYDEVICE).into()),
            5 => Ok(Tensor::<NdArray, 5>::zeros(shape, &NDARRAYDEVICE).into()),
            _ => Err(
                TensorError::WrongDimensions.into(), /*("Unsupported dimensions")*/
            ),
        }
    }

    /// Creates a tensor whose shape and dimensions is similar to the one in use
    fn zeros_like(&self) -> PyResult<Self> {
        match self {
            TensorPy::TensorOne(val) => Ok(val.inner.zeros_like().into()),

            TensorPy::TensorTwo(val) => Ok(val.inner.zeros_like().into()),

            TensorPy::TensorThree(val) => Ok(val.inner.zeros_like().into()),

            TensorPy::TensorFour(val) => Ok(val.inner.zeros_like().into()),

            TensorPy::TensorFive(val) => Ok(val.inner.zeros_like().into()),
            _ => Err(TensorError::NonApplicableMethod.into()),
        }
    }
}

impl_tensor_conversions_ndarray!(
    Tensor1,
    Tensor1Bool,
    1,
    TensorOne,
    TensorOneBool,
    TensorOneInt,
    Tensor1Int
);
impl_tensor_conversions_ndarray!(
    Tensor2,
    Tensor2Bool,
    2,
    TensorTwo,
    TensorTwoBool,
    TensorTwoInt,
    Tensor2Int
);
impl_tensor_conversions_ndarray!(
    Tensor3,
    Tensor3Bool,
    3,
    TensorThree,
    TensorThreeBool,
    TensorThreeInt,
    Tensor3Int
);
impl_tensor_conversions_ndarray!(
    Tensor4,
    Tensor4Bool,
    4,
    TensorFour,
    TensorFourBool,
    TensorFourInt,
    Tensor4Int
);
impl_tensor_conversions_ndarray!(
    Tensor5,
    Tensor5Bool,
    5,
    TensorFive,
    TensorFiveBool,
    TensorFiveInt,
    Tensor5Int
);

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
