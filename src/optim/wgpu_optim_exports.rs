//! Optim module written targeting the Wgpu backend

use pyo3::prelude::*;
use burn::{optim::*, backend::wgpu::*};
use crate::{implement_wgpu_interface, implement_send_and_sync, for_normal_struct_enums};
use super::common_exports;


// Implement the types given the signature is that of two parameters
// implement_wgpu_interface!(AdaGradStatePy, AdaGradState, "AdaGrad state");
// implement_wgpu_interface!(AdamStatePy, AdamState, "Adam State");

// macro_rules! implement_type_with_two_const_requirements {
//     ($name:ident, $type: ident, $val:expr) => {
//         #[pyclass]
//         pub struct $name {
//             pub inner: $type<Wgpu, $val>
//         }
//     }
// }

// fn generate_adagradstate_given_value(val: usize) {
//     implement_type_with_two_const_requirements!(AdaGradStatePy, AdaGradState, {val});
// }