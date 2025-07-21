//! Optim module written targeting the Wgpu backend

use pyo3::prelude::*;
use burn::{optim::*, backend::wgpu::*};
use crate::{implement_wgpu_interface, implement_send_and_sync, for_normal_struct_enums};
use super::common_exports;


// Implement the types given the signature is that of two parameters
// implement_wgpu_interface!(AdaGradStatePy, AdaGradState, "AdaGrad state");
// implement_wgpu_interface!(AdamStatePy, AdamState, "Adam State");