//! Common Exports of the optim module regardless of backend

use burn::{prelude::*, optim::*};
use pyo3::prelude::*;
use crate::{for_normal_struct_enums, implement_send_and_sync};

for_normal_struct_enums!(AdaGradPy, AdaGrad);
for_normal_struct_enums!(AdaGradConfigPy, AdaGradConfig);
for_normal_struct_enums!(AdamPy, Adam);
for_normal_struct_enums!(AdamConfigPy, AdamConfig);
for_normal_struct_enums!(AdamWPy, AdamW);
for_normal_struct_enums!(AdamWConfigPy, AdamWConfig);
for_normal_struct_enums!(GradientsParamsPy, GradientsParams);
for_normal_struct_enums!(RmsPropPy, RmsProp);
for_normal_struct_enums!(RmsPropConfigPy, RmsPropConfig);
for_normal_struct_enums!(RmsPropMomentumPy, RmsPropMomentum);
for_normal_struct_enums!(SgdConfigPy, SgdConfig);

implement_send_and_sync!(GradientsParamsPy);

pub mod decay_exports {
    use super::*;
    use super::decay::*;
    for_normal_struct_enums!(WeightDecayPy, WeightDecay);
    for_normal_struct_enums!(WeightDecayConfigPy, WeightDecayConfig);
   
}