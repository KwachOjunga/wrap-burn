#![allow(unused)]
#![recursion_limit = "512"]

//! [`pyburn`] attempts to expose burn's modules and methods in a manner that permits it to work
//! as a python interface. This module exposes the [`burn::nn`] module.

use crate::tensor::tensor_error::TensorError;
use crate::{
    for_normal_struct_enums, implement_ndarray_interface, implement_send_and_sync,
    implement_wgpu_interface,
};
use burn::backend::ndarray::NdArrayDevice;
use burn::backend::wgpu::WgpuDevice;
use burn::nn::Linear;
use burn::nn::*;
use burn::prelude::*;
use pyo3::prelude::*;

pub mod common_nn_exports;
// mod ndarray_nn_exports;
// mod wgpu_nn_exports;
// I thought send and Sync were implemented automatically??

pub static WGPUDEVICE: WgpuDevice = WgpuDevice::DefaultDevice;
pub static NDARRAYDEVICE: NdArrayDevice = NdArrayDevice::Cpu;
