#![allow(unused)]

mod ndarray_optim_exports;
mod wgpu_optim_exports; 
mod common_exports;

use burn::optim::*;
use burn::prelude::*;
use pyo3::prelude::*;



#[cfg(feature = "wgpu")]
pub mod wgpu {}

#[cfg(feature = "ndarray")]
pub mod ndarray {}
