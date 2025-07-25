#![recursion_limit = "256"]

use pyo3::prelude::*;

mod nn;
mod optim;
mod record;
mod tensor;
mod train;

#[macro_export]
macro_rules! implement_ndarray_interface {
    ($(#[$meta:meta])* $name:ident, $actual_type:ident ,$doc:literal) => {
        use burn::backend::ndarray::*;

        #[doc = $doc]
        #[pyclass]
        pub struct $name {
            pub inner: $actual_type<NdArray>,
        }
    };

    ($(#[$meta:meta])* $name:ident, $actual_type:ident) => {
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
    ($(#[$meta:meta])* $name:ident, $actual_type:ident, $doc:literal) => {
        use burn::backend::wgpu::*;
        #[doc = $doc]
        #[pyclass]
        pub struct $name {
            pub inner: $actual_type<Wgpu>,
        }
    };

    ($(#[$meta:meta])* $name:ident, $actual_type:ident) => {
        use burn::backend::wgpu::*;

        #[pyclass]
        pub struct $name {
            pub inner: $actual_type<Wgpu>,
        }
    };
}

#[macro_export]
macro_rules! for_normal_struct_enums {
    ($(#[$meta:meta])* $name:ident, $actual_type:ident, $doc:literal) => {
        #[derive(Clone)]
        #[doc = $doc]
        #[pyclass]
        pub struct $name(pub $actual_type);

        impl From<$name> for $actual_type {
            fn from(other: $name) -> Self {
                other.0
            }
        }
    };

    ($(#[$meta:meta])* $name:ident, $actual_type:ident) => {
        #[pyclass]
        pub struct $name(pub $actual_type);

        impl From<$name> for $actual_type {
            fn from(other: $name) -> Self {
                other.0
            }
        }
    };
}

#[cfg(feature = "wgpu")]
#[pymodule]
mod pyburn {
    // use super::*;

    #[pymodule_export]
    use super::nn::wgpu_nn;
}

#[cfg(feature = "ndarray")]
#[pymodule]
mod pyburn {

    #[pymodule_export]
    use super::nn::ndarray as nn;
}
// pub use optim::ndarray as optim;
