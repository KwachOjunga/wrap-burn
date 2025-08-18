//! Implment burn::module 
//! 
//! Some of the structs implemented heer are necessary for 
//! operations like quantization. This enables a user to opt in
//! to such features if they wish`to.
//! 
//! Note: This is untested code, so it may not work as expected.

// [TODO]: Implement this crate's function's in a capacity that illustrates its utility.


use crate::{for_normal_struct_enums, implement_send_and_sync};
use burn::module::*;
use pyo3::prelude::*;   
