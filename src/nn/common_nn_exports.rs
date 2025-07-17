use crate::{for_normal_struct_enums};
use burn::nn::*;
use burn::prelude::*;
use pyo3::prelude::*;

pub mod pool_exports {
    pub(crate) use super::*;
    use burn::nn::pool::*;

    /// This is  the typical AdaptivePool1d layer
    for_normal_struct_enums!(PyAdaptiveAvgPool1d, AdaptiveAvgPool1d);
    for_normal_struct_enums!(PyAdaptiveAvgPool1dConfig, AdaptiveAvgPool1dConfig);
    for_normal_struct_enums!(PyAdaptiveAvgPool2d, AdaptiveAvgPool2d);
    for_normal_struct_enums!(PyAdaptiveAvgPool2dConfig, AdaptiveAvgPool2dConfig);
    for_normal_struct_enums!(PyAvgPool1d, AvgPool1d);
    for_normal_struct_enums!(PyAvgPool1dConfig, AvgPool1dConfig);
    for_normal_struct_enums!(PyAvgPool2d, AvgPool2d);
    for_normal_struct_enums!(PyAvgPool2dConfig, AvgPool2dConfig);
    for_normal_struct_enums!(PyMaxPool1d, MaxPool1d);
    for_normal_struct_enums!(PyMaxPool1dConfig, MaxPool1dConfig);
    for_normal_struct_enums!(PyMaxPool2d, MaxPool2d);
    for_normal_struct_enums!(PyMaxPool2dConfig, MaxPool2dConfig);
}