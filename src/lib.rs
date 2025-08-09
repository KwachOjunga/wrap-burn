#![recursion_limit = "256"]

use pyo3::prelude::*;

pub mod nn;
pub mod optim;
mod record;
pub mod tensor;
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

#[pymodule]
pub mod pyburn {

    use super::*;

    /// Modules built for the wgpu backend
    #[cfg(feature = "wgpu")]
    #[pymodule]
    mod wgpu {

        /// Train module
        #[pymodule_export]
        use super::train::wgpu_train;

        /// Neural network module
        #[pymodule_export]
        use super::nn::wgpu_nn;

        /// Basic Tensor module with wgpu as its backend
        #[pymodule_export]
        use super::tensor::wgpu_tensor;

        /// Optimization module for wgpu backend
        #[pymodule_export]
        use super::optim::wgpu_optim;
    }

    /// Modules built for the ndarray backend
    #[cfg(feature = "ndarray")]
    #[pymodule]
    mod ndarray {

        /// Train module
        #[pymodule_export]
        use super::train::ndarray_train;

        /// Neural network module
        #[pymodule_export(name = "ndarray_nn")]
        use super::nn::ndarray_nn;

        /// Basic tensor module with the cpu as its backend
        #[pymodule_export]
        use super::tensor::ndarray_tensor;

        /// Optimization module for ndarray backend
        #[pymodule_export]
        use super::optim::ndarray_optim;
    }
}
