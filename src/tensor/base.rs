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
    TensorFiveBool(Tensor5Bool)
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
    fn abs(&self) -> Self {
        match self {
            TensorPy::TensorOne(val) => Into::<TensorPy>::into(val.inner.clone().abs()),
            TensorPy::TensorTwo(val) => Into::<TensorPy>::into(val.inner.clone().abs()),
            TensorPy::TensorThree(val) => Into::<TensorPy>::into(val.inner.clone().abs()),
            TensorPy::TensorFour(val) => Into::<TensorPy>::into(val.inner.clone().abs()),
            TensorPy::TensorFive(val) => Into::<TensorPy>::into(val.inner.clone().abs()),
        }
    }

    fn add(&self, other: TensorPy) -> Self {
        match self {
            TensorPy::TensorOne(val) => Into::<TensorPy>::into(
                val.inner.clone().add(
                    Into::<anyhow::Result<Tensor<Wgpu, 1>>>::into(other)
                        .expect("expected 1 dim tensor"),
                ),
            ),
            TensorPy::TensorTwo(val) => Into::<TensorPy>::into(
                val.inner.clone().add(
                    Into::<anyhow::Result<Tensor<Wgpu, 2>>>::into(other)
                        .expect("expected 2 dim tensor"),
                ),
            ),
            TensorPy::TensorThree(val) => Into::<TensorPy>::into(
                val.inner.clone().add(
                    Into::<anyhow::Result<Tensor<Wgpu, 3>>>::into(other)
                        .expect("expected 3 dim tensor"),
                ),
            ),
            TensorPy::TensorFour(val) => Into::<TensorPy>::into(
                val.inner.clone().add(
                    Into::<anyhow::Result<Tensor<Wgpu, 4>>>::into(other)
                        .expect("expected 4 dim tensor"),
                ),
            ),
            TensorPy::TensorFive(val) => Into::<TensorPy>::into(
                val.inner.clone().add(
                    Into::<anyhow::Result<Tensor<Wgpu, 5>>>::into(other)
                        .expect("expected 5 dim tensor"),
                ),
            ),
        }
    }

    fn add_scalar(&self, input: f32) -> Self {
        match self {
            TensorPy::TensorOne(val) => Into::<TensorPy>::into(val.inner.clone().add_scalar(input)),
            TensorPy::TensorTwo(val) => Into::<TensorPy>::into(val.inner.clone().add_scalar(input)),
            TensorPy::TensorThree(val) => {
                Into::<TensorPy>::into(val.inner.clone().add_scalar(input))
            }
            TensorPy::TensorFour(val) => {
                Into::<TensorPy>::into(val.inner.clone().add_scalar(input))
            }
            TensorPy::TensorFive(val) => {
                Into::<TensorPy>::into(val.inner.clone().add_scalar(input))
            }
        }
    }

    fn sub(&self, other: TensorPy) -> Self {
        match self {
            TensorPy::TensorOne(val) => Into::<TensorPy>::into(
                val.inner.clone().sub(
                    Into::<anyhow::Result<Tensor<Wgpu, 1>>>::into(other)
                        .expect("expected 1 dim tensor"),
                ),
            ),
            TensorPy::TensorTwo(val) => Into::<TensorPy>::into(
                val.inner.clone().sub(
                    Into::<anyhow::Result<Tensor<Wgpu, 2>>>::into(other)
                        .expect("expected 2 dim tensor"),
                ),
            ),
            TensorPy::TensorThree(val) => Into::<TensorPy>::into(
                val.inner.clone().sub(
                    Into::<anyhow::Result<Tensor<Wgpu, 3>>>::into(other)
                        .expect("expected 3 dim tensor"),
                ),
            ),
            TensorPy::TensorFour(val) => Into::<TensorPy>::into(
                val.inner.clone().sub(
                    Into::<anyhow::Result<Tensor<Wgpu, 4>>>::into(other)
                        .expect("expected 4 dim tensor"),
                ),
            ),
            TensorPy::TensorFive(val) => Into::<TensorPy>::into(
                val.inner.clone().sub(
                    Into::<anyhow::Result<Tensor<Wgpu, 5>>>::into(other)
                        .expect("expected 5 dim tensor"),
                ),
            ),
        }
    }

    fn sub_scalar(&self, input: f32) -> Self {
        match self {
            TensorPy::TensorOne(val) => Into::<TensorPy>::into(val.inner.clone().sub_scalar(input)),
            TensorPy::TensorTwo(val) => Into::<TensorPy>::into(val.inner.clone().sub_scalar(input)),
            TensorPy::TensorThree(val) => {
                Into::<TensorPy>::into(val.inner.clone().sub_scalar(input))
            }
            TensorPy::TensorFour(val) => {
                Into::<TensorPy>::into(val.inner.clone().sub_scalar(input))
            }
            TensorPy::TensorFive(val) => {
                Into::<TensorPy>::into(val.inner.clone().sub_scalar(input))
            }
        }
    }

    /// Test if any element in the Tensor evaluates to True
    fn any(&self) -> Self {
        match self {
            TensorPy::TensorOne(val) => Into::<TensorPy>::into(val.inner.clone().any()),
            TensorPy::TensorTwo(val) => Into::<TensorPy>::into(val.inner.clone().any()),
            TensorPy::TensorThree(val) => Into::<TensorPy>::into(val.inner.clone().any()),
            TensorPy::TensorFour(val) => Into::<TensorPy>::into(val.inner.clone().any()),
            TensorPy::TensorFive(val) => Into::<TensorPy>::into(val.inner.clone().any()),
        }
    }

    fn all(&self) -> Self {
        match self {
            TensorPy::TensorOne(val) => Into::<TensorPy>::into(val.inner.clone().all()),
            TensorPy::TensorTwo(val) => Into::<TensorPy>::into(val.inner.clone().all()),
            TensorPy::TensorThree(val) => Into::<TensorPy>::into(val.inner.clone().all()),
            TensorPy::TensorFour(val) => Into::<TensorPy>::into(val.inner.clone().all()),
            TensorPy::TensorFive(val) => Into::<TensorPy>::into(val.inner.clone().all()),
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
