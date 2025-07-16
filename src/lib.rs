mod nn;
mod optim;
mod train;
mod record;

#[macro_export]
macro_rules! implement_ndarray_interface {
    ($name:ident, $actual_type:ident) => {
        use burn::backend::ndarray::*;

        #[pyclass]
        pub struct $name {
            pub inner: $actual_type<NdArray>,
        }
    };
}

#[macro_export]
macro_rules! implement_send_and_sync {
    ($name:ty) => {
        unsafe impl Send for $name {}
        unsafe impl Sync for $name {}
    };
}

#[macro_export]
macro_rules! implement_wgpu_interface {
    ($name:ident, $actual_type:ident) => {
        use burn::backend::wgpu::*;

        #[pyclass]
        pub struct $name {
            pub inner: $actual_type<Wgpu>,
        }
    };
}

#[macro_export]
macro_rules! for_normal_struct_enums {
    ($name:ident, $actual_type:ident) => {
        #[pyclass]
        pub struct $name(pub $actual_type);
    };
}



#[cfg(feature = "wgpu")]
pub use nn::wgpu as nn;
// pub use optim::wgpu as optim;

#[cfg(feature = "ndarray")]
pub use nn::ndarray as nn;
// pub use optim::ndarray as optim;
