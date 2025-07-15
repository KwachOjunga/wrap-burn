mod nn;
mod optim;

#[cfg(feature = "wgpu")]
pub use nn::wgpu as py_nn;
pub use optim::wgpu as py_optim;

#[cfg(feature = "ndarray")]
pub use nn::ndarray as py_nn;
pub use optim::ndarray as py_optim;
