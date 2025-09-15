//! This implements burn::module
//!
//! Some of the structs implemented here are necessary for
//! operations like quantization. This enables a user to opt in
//! to such features if they wish`to.
//!
//! Note: This is untested code, so it may not work as expected.

// [TODO]: Implement this crate's function's in a capacity that illustrates its utility.

use crate::for_normal_struct_enums;
use burn::module::*;
use pyo3::prelude::*;

for_normal_struct_enums!(AttributePy, Attribute);
for_normal_struct_enums!(ConstantRecordPy, ConstantRecord);
for_normal_struct_enums!(ContentPy, Content);
for_normal_struct_enums!(DisplaySettingsPy, DisplaySettings);

for_normal_struct_enums!(QuantizerPy, Quantizer);
