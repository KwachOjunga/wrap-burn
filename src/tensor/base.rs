//! Warning. The current implementation of TensorPy is grossly inefficient.

use std::f32;

// use std::sync::{Arc, Mutex};
use super::tensor_error::*;
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

// impl TensorPy {
//     fn inner<T: Backend>(&self) -> T {
//         match self {
//             TensorPy::TensorOne(val) => val.inner,
//             TensorPy::TensorTwo(val) => val.inner,
//             TensorPy::TensorThree(val) => val.inner,
//             TensorPy::TensorFour(val) => val.inner,
//             TensorPy::TensorFive(val) => val.inner,
//         }
//     }
// }

// -> Initial method val.inner.clone().abs()

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

// Conversion between TensorPy and various types.

impl From<Tensor<Wgpu, 1>> for Tensor1 {
    fn from(other: Tensor<Wgpu, 1>) -> Self {
        Self { inner: other }
    }
}

impl From<Tensor<Wgpu, 1>> for TensorPy {
    fn from(other: Tensor<Wgpu, 1>) -> Self {
        Self::TensorOne(other.into())
    }
}

impl From<TensorPy> for anyhow::Result<Tensor<Wgpu, 1>> {
    fn from(other: TensorPy) -> anyhow::Result<Tensor<Wgpu, 1>> {
        match other {
            TensorPy::TensorOne(val) => Ok(val.inner),
            _ => Err(WrongDimensions.into()),
        }
    }
}

impl From<TensorPy> for anyhow::Result<Tensor<Wgpu, 1, Bool>> {
    fn from(other: TensorPy) -> anyhow::Result<Tensor<Wgpu, 1, Bool>> {
        match other {
            TensorPy::TensorOneBool(val) => Ok(val.inner),
            _ => Err(WrongDimensions.into()),
        }
    }
}

impl From<Tensor<Wgpu, 1, Bool>> for TensorPy {
    fn from(other: Tensor<Wgpu, 1, Bool>) -> Self {
        Self::TensorOneBool(Tensor1Bool { inner: other })
    }
}

impl From<Tensor<Wgpu, 2>> for Tensor2 {
    fn from(other: Tensor<Wgpu, 2>) -> Self {
        Self { inner: other }
    }
}

impl From<Tensor<Wgpu, 2>> for TensorPy {
    fn from(other: Tensor<Wgpu, 2>) -> Self {
        Self::TensorTwo(other.into())
    }
}

impl From<TensorPy> for anyhow::Result<Tensor<Wgpu, 2>> {
    fn from(other: TensorPy) -> anyhow::Result<Tensor<Wgpu, 2>> {
        match other {
            TensorPy::TensorTwo(val) => Ok(val.inner),
            _ => Err(WrongDimensions.into()),
        }
    }
}

impl From<Tensor<Wgpu, 2, Bool>> for TensorPy {
    fn from(other: Tensor<Wgpu, 2, Bool>) -> Self {
        Self::TensorTwoBool(Tensor2Bool { inner: other })
    }
}

// 3 dim Tensor
impl From<Tensor<Wgpu, 3>> for Tensor3 {
    fn from(other: Tensor<Wgpu, 3>) -> Self {
        Self { inner: other }
    }
}

impl From<Tensor<Wgpu, 3>> for TensorPy {
    fn from(other: Tensor<Wgpu, 3>) -> Self {
        Self::TensorThree(other.into())
    }
}

impl From<TensorPy> for anyhow::Result<Tensor<Wgpu, 3>> {
    fn from(other: TensorPy) -> anyhow::Result<Tensor<Wgpu, 3>> {
        match other {
            TensorPy::TensorThree(val) => Ok(val.inner),
            _ => Err(WrongDimensions.into()),
        }
    }
}

impl From<Tensor<Wgpu, 3, Bool>> for TensorPy {
    fn from(other: Tensor<Wgpu, 3, Bool>) -> Self {
        Self::TensorThreeBool(Tensor3Bool { inner: other })
    }
}

// 4 dim Tensor
impl From<Tensor<Wgpu, 4>> for Tensor4 {
    fn from(other: Tensor<Wgpu, 4>) -> Self {
        Self { inner: other }
    }
}

impl From<Tensor<Wgpu, 4>> for TensorPy {
    fn from(other: Tensor<Wgpu, 4>) -> Self {
        Self::TensorFour(other.into())
    }
}

impl From<TensorPy> for anyhow::Result<Tensor<Wgpu, 4>> {
    fn from(other: TensorPy) -> anyhow::Result<Tensor<Wgpu, 4>> {
        match other {
            TensorPy::TensorFour(val) => Ok(val.inner),
            _ => Err(WrongDimensions.into()),
        }
    }
}

impl From<Tensor<Wgpu, 4, Bool>> for TensorPy {
    fn from(other: Tensor<Wgpu, 4, Bool>) -> Self {
        Self::TensorFourBool(Tensor4Bool { inner: other })
    }
}

// 5 dim Tensor
impl From<Tensor<Wgpu, 5>> for Tensor5 {
    fn from(other: Tensor<Wgpu, 5>) -> Self {
        Self { inner: other }
    }
}

impl From<Tensor<Wgpu, 5>> for TensorPy {
    fn from(other: Tensor<Wgpu, 5>) -> Self {
        Self::TensorFive(other.into())
    }
}

impl From<TensorPy> for anyhow::Result<Tensor<Wgpu, 5>> {
    fn from(other: TensorPy) -> anyhow::Result<Tensor<Wgpu, 5>> {
        match other {
            TensorPy::TensorFive(val) => Ok(val.inner),
            _ => Err(WrongDimensions.into()),
        }
    }
}

impl From<Tensor<Wgpu, 5, Bool>> for TensorPy {
    fn from(other: Tensor<Wgpu, 5, Bool>) -> Self {
        Self::TensorFiveBool(Tensor5Bool { inner: other })
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
