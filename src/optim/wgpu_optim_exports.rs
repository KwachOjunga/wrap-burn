use crate::{/*for_normal_struct_enums, implement_send_and_sync,*/ implement_wgpu_interface};
use burn::optim::Sgd;
use burn::optim::momentum::*;

use pyo3::prelude::*;

implement_wgpu_interface!(MomentumPy, Momentum);

implement_wgpu_interface!(SdgPy, Sgd);

// [TODO:] @kwach fix this
// #[pymethods]
// impl SgdPy {
//     #[new]
//     fn new(weight_decay: Option<crate::optim::common_exports::decay_exports::WeightDecayConfigPy>, momentum: Option<crate::optim::common_exports::momentum::MomentumConfigPy>) -> Self {
//         let momentum_config = match momentum {
//             Some(m) => Some(m.0),
//             None => None,
//         };
//         let weight_decay_config = match weight_decay {
//             Some(w) => Some(w.0),
//             None => None,
//         };
//         let config = burn::optim::SgdConfig::new().with_weight_decay(Some(weight_decay_config.0)).with_momentum(momentum_config.0).with_gradient_clipping(Some()); {
//             weight_decay: weight_decay_config,
//             momentum: momentum_config,
//         };
//         SgdPy {
//             inner: Sgd::new(config),
//         }
//     }
// }
