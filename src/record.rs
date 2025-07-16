#![allow(unused)]

use burn::record::*;
use pyo3::{exceptions::PyValueError, prelude::*};

#[pyclass]
pub struct PyRecorderError {
    pub inner: RecorderError,
}

impl From<PyRecorderError> for PyErr {
    fn from(other: PyRecorderError) -> Self {
        match other.inner {
            RecorderError::FileNotFound(_) => PyValueError::new_err("File not found"),
            RecorderError::DeserializeError(_) => {
                PyValueError::new_err("Failed to deserialize record")
            }
            RecorderError::Unknown(_) => {
                PyValueError::new_err("Sorry! an unknown error has occurred")
            }
        }
    }
}

impl From<RecorderError> for PyRecorderError {
    fn from(other: RecorderError) -> Self {
        Self { inner: other }
    }
}
