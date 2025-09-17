#![recursion_limit = "256"]

use pyo3::prelude::*;

mod grad_clipping;
mod lr_scheduler;
mod module;
pub mod nn;
pub mod optim;
mod record;
pub mod tensor;
mod tests;
mod train;
// use tensor::wgpu_base::*;

#[macro_export]
macro_rules! impl_tensor_conversions_wgpu {
    (
        $tensor_ty:ident, $tensor_bool_ty:ident, $dim:expr, $variant:ident, $variant_bool:ident, $variant_int:ident, $tensor_int_ty:ident
    ) => {
        // Tensor<Wgpu, N> -> Wrapper
        impl From<Tensor<Wgpu, $dim>> for $tensor_ty {
            fn from(other: Tensor<Wgpu, $dim>) -> Self {
                Self { inner: other }
            }
        }
        // Tensor<Wgpu, N> -> TensorPy
        impl From<Tensor<Wgpu, $dim>> for TensorPy {
            fn from(other: Tensor<Wgpu, $dim>) -> Self {
                TensorPy::$variant(other.into())
            }
        }
        // TensorPy -> anyhow::Result<Tensor<Wgpu, N>>
        impl From<TensorPy> for anyhow::Result<Tensor<Wgpu, $dim>> {
            fn from(other: TensorPy) -> anyhow::Result<Tensor<Wgpu, $dim>> {
                match other {
                    TensorPy::$variant(val) => Ok(val.inner),
                    _ => Err(WrongDimensions.into()),
                }
            }
        }
        // Tensor<Wgpu, N, Bool> -> WrapperBool
        impl From<Tensor<Wgpu, $dim, Bool>> for $tensor_bool_ty {
            fn from(other: Tensor<Wgpu, $dim, Bool>) -> Self {
                Self { inner: other }
            }
        }
        // Tensor<Wgpu, N, Bool> -> TensorPy
        impl From<Tensor<Wgpu, $dim, Bool>> for TensorPy {
            fn from(other: Tensor<Wgpu, $dim, Bool>) -> Self {
                TensorPy::$variant_bool($tensor_bool_ty { inner: other })
            }
        }
        // TensorPy -> anyhow::Result<Tensor<Wgpu, N, Bool>>
        impl From<TensorPy> for anyhow::Result<Tensor<Wgpu, $dim, Bool>> {
            fn from(other: TensorPy) -> anyhow::Result<Tensor<Wgpu, $dim, Bool>> {
                match other {
                    TensorPy::$variant_bool(val) => Ok(val.inner),
                    _ => Err(WrongDimensions.into()),
                }
            }
        }
        // Wrapper -> Tensor<Wgpu, N>
        impl From<$tensor_ty> for Tensor<Wgpu, $dim> {
            fn from(other: $tensor_ty) -> Self {
                other.inner
            }
        }
        // WrapperBool -> Tensor<Wgpu, N, Bool>
        impl From<$tensor_bool_ty> for Tensor<Wgpu, $dim, Bool> {
            fn from(other: $tensor_bool_ty) -> Self {
                other.inner
            }
        }

        // Tensor<Wgpu, N, Int> -> TensorPy
        impl From<Tensor<Wgpu, $dim, Int>> for TensorPy {
            fn from(other: Tensor<Wgpu, $dim, Int>) -> Self {
                TensorPy::$variant_int($tensor_int_ty { inner: other })
            }
        }
        // TensorPy -> anyhow::Result<Tensor<Wgpu, N, Int>>
        impl From<TensorPy> for anyhow::Result<Tensor<Wgpu, $dim, Int>> {
            fn from(other: TensorPy) -> anyhow::Result<Tensor<Wgpu, $dim, Int>> {
                match other {
                    TensorPy::$variant_int(val) => Ok(val.inner),
                    _ => Err(WrongDimensions.into()),
                }
            }
        }

        // WrapperInt -> Tensor<Wgpu, N, Int>
        impl From<$tensor_int_ty> for Tensor<Wgpu, $dim, Int> {
            fn from(other: $tensor_int_ty) -> Self {
                other.inner
            }
        }
    };
}

#[macro_export]
macro_rules! impl_tensor_conversions_ndarray {
    (
        $tensor_ty:ident, $tensor_bool_ty:ident, $dim:expr, $variant:ident, $variant_bool:ident, $variant_int:ident, $tensor_int_ty:ident
    ) => {
        // use super::tensor::NdArray;

        // Tensor<NdArray, N> -> Wrapper
        impl From<Tensor<NdArray, $dim>> for $tensor_ty {
            fn from(other: Tensor<NdArray, $dim>) -> Self {
                Self { inner: other }
            }
        }
        // Tensor<NdArray, N> -> TensorPy
        impl From<Tensor<NdArray, $dim>> for TensorPy {
            fn from(other: Tensor<NdArray, $dim>) -> Self {
                TensorPy::$variant(other.into())
            }
        }
        // TensorPy -> anyhow::Result<Tensor<NdArray, N>>
        impl From<TensorPy> for anyhow::Result<Tensor<NdArray, $dim>> {
            fn from(other: TensorPy) -> anyhow::Result<Tensor<NdArray, $dim>> {
                match other {
                    TensorPy::$variant(val) => Ok(val.inner),
                    _ => Err(WrongDimensions.into()),
                }
            }
        }
        // Tensor<NdArray, N, Bool> -> WrapperBool
        impl From<Tensor<NdArray, $dim, Bool>> for $tensor_bool_ty {
            fn from(other: Tensor<NdArray, $dim, Bool>) -> Self {
                Self { inner: other }
            }
        }
        // Tensor<NdArray, N, Bool> -> TensorPy
        impl From<Tensor<NdArray, $dim, Bool>> for TensorPy {
            fn from(other: Tensor<NdArray, $dim, Bool>) -> Self {
                TensorPy::$variant_bool($tensor_bool_ty { inner: other })
            }
        }
        // TensorPy -> anyhow::Result<Tensor<NdArray, N, Bool>>
        impl From<TensorPy> for anyhow::Result<Tensor<NdArray, $dim, Bool>> {
            fn from(other: TensorPy) -> anyhow::Result<Tensor<NdArray, $dim, Bool>> {
                match other {
                    TensorPy::$variant_bool(val) => Ok(val.inner),
                    _ => Err(WrongDimensions.into()),
                }
            }
        }
        // Wrapper -> Tensor<NdArray, N>
        impl From<$tensor_ty> for Tensor<NdArray, $dim> {
            fn from(other: $tensor_ty) -> Self {
                other.inner
            }
        }
        // WrapperBool -> Tensor<NdArray, N, Bool>
        impl From<$tensor_bool_ty> for Tensor<NdArray, $dim, Bool> {
            fn from(other: $tensor_bool_ty) -> Self {
                other.inner
            }
        }
        // Tensor<NdArray, N, Int> -> TensorPy
        impl From<Tensor<NdArray, $dim, Int>> for TensorPy {
            fn from(other: Tensor<NdArray, $dim, Int>) -> Self {
                TensorPy::$variant_int($tensor_int_ty { inner: other })
            }
        }
        // TensorPy -> anyhow::Result<Tensor<NdArray, N, Int>>
        impl From<TensorPy> for anyhow::Result<Tensor<NdArray, $dim, Int>> {
            fn from(other: TensorPy) -> anyhow::Result<Tensor<NdArray, $dim, Int>> {
                match other {
                    TensorPy::$variant_int(val) => Ok(val.inner),
                    _ => Err(WrongDimensions.into()),
                }
            }
        }

        // WrapperInt -> Tensor<NdArray, N, Int>
        impl From<$tensor_int_ty> for Tensor<NdArray, $dim, Int> {
            fn from(other: $tensor_int_ty) -> Self {
                other.inner
            }
        }
    };
}

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
        // use burn::backend::wgpu::*;
        #[doc = $doc]
        #[pyclass]
        // #[derive(Clone)]
        pub struct $name {
            pub inner: $actual_type<Wgpu>,
        }
    };

    ($(#[$meta:meta])* $name:ident, $actual_type:ident) => {
        use burn::backend::wgpu::*;

        #[pyclass]
        // #[derive(Clone)]
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

        impl From<$actual_type> for $name {
            fn from(other: $actual_type) -> Self {
                Self(other)
            }
        }
    };

    ($(#[$meta:meta])* $name:ident, $actual_type:ident) => {
        #[pyclass]
        pub struct $name(pub $actual_type);

        impl From<$actual_type> for $name {
            fn from(other: $actual_type) -> Self {
                Self(other)
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

        #[pymodule_export]
        use super::module::module;

        #[pymodule_export]
        use super::lr_scheduler::scheduler;

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

        #[pymodule_export]
        use super::module::module;

        #[pymodule_export]
        use super::lr_scheduler::scheduler;

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
