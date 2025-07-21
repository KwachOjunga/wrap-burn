use pyo3::{ exceptions::PyValueError, prelude::*};

#[pyclass]
#[doc = "Tensor Error: to be used when raisig exceptions that involve tensors"]
pub enum TensorError {
    WrongDimensions
}

impl From<TensorError> for PyErr {
    fn from(other: TensorError) -> Self {
        match other {
            TensorError::WrongDimensions => PyValueError::new_err("Check input tensor dimensions")
        }
    }
}