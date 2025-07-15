#![allow(unused)]

use burn::optim::*;
use burn::prelude::*;
use pyo3::prelude::*;

#[cfg(feature = "wgpu")]
pub mod wgpu {}

#[cfg(feature = "ndarray")]
pub mod ndarray {}
