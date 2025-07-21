use pyo3::{exceptions::PyValueError, prelude::*};

#[pyclass]
#[doc = "Tensor Error: to be used when raisig exceptions that involve tensors"]
#[non_exhaustive]
pub enum TensorError {
    WrongDimensions,
}

impl From<TensorError> for PyErr {
    fn from(other: TensorError) -> Self {
        match other {
            TensorError::WrongDimensions => PyValueError::new_err("Check input tensor dimensions"),
        }
    }
}
