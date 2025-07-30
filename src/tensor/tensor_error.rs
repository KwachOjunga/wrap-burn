use core::error::Error;
use core::fmt;
use pyo3::{exceptions::PyValueError, prelude::*};

/// Container that serves to hold errors of tensor operations
/// It's the primary wrapper that allows exceptions to be raised from tensor errors
#[pyclass]
#[derive(Debug)]
#[doc = "Tensor Error: to be used when raising exceptions that involve tensors"]
#[non_exhaustive]
pub enum TensorError {
    WrongDimensions,
    NonApplicableMethod,
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TensorError!")
    }
}

#[derive(Debug)]
pub struct NonApplicableMethod;

impl fmt::Display for NonApplicableMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "The method does not apply to a tensor of this nature!")
    }
}

impl Error for NonApplicableMethod {}

#[derive(Debug)]
pub struct WrongDimensions;

impl fmt::Display for WrongDimensions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Check input dimensions!")
    }
}

impl Error for WrongDimensions {}

impl From<TensorError> for PyErr {
    fn from(other: TensorError) -> Self {
        match other {
            TensorError::WrongDimensions => PyValueError::new_err("Check input tensor dimensions"),
            TensorError::NonApplicableMethod => {
                PyValueError::new_err("Method does not apply to this Tensor")
            }
        }
    }
}
