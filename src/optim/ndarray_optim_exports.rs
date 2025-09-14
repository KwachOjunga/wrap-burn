use crate::implement_ndarray_interface;
use burn::optim::Sgd;
use pyo3::prelude::*;

implement_ndarray_interface!(SgdPy, Sgd);
