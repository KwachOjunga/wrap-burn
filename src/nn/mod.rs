#![allow(unused)]
#![recursion_limit = "512"]

//! [`pyburn`] attempts to expose burn's modules and methods in a manner that permits it to work
//! as a python interface. This module exposes the [`burn::nn`] module.

use burn::backend::ndarray::NdArrayDevice;
use burn::backend::wgpu::WgpuDevice;
// I thought send and Sync were implemented automatically??

pub static WGPUDEVICE: WgpuDevice = WgpuDevice::DefaultDevice;
pub static NDARRAYDEVICE: NdArrayDevice = NdArrayDevice::Cpu;
