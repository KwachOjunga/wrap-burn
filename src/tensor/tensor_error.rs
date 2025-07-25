use core::error::Error;
use core::fmt;
use pyo3::{exceptions::PyValueError, prelude::*};

/// Container that serves to hold errors of tensor operations
/// It's the primary wrapper that allows exceptions to be raised from tensor errors
#[pyclass]
#[derive(Debug)]
#[doc = "Tensor Error: to be used when raisig exceptions that involve tensors"]
#[non_exhaustive]
pub enum TensorError {
    WrongDimensions,
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TensorError!")
    }
}

// impl Error for TensorError {
//     fn source(&self) -> Option<&(dyn Error + 'static)> {
//         match self {
//             TensorError::WrongDimensions => Some(&<WrongDimensions as Error>::source())
//         }
//     }
// }

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
        }
    }
}
