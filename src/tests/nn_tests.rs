#![allow(unused)]

#[cfg(test)]
mod nn_test {
    use crate::tensor::ndarray_base;
    use crate::tensor::wgpu_base;
    use burn::backend::NdArray;
    use dhat;

    #[global_allocator]
    static ALLOC: dhat::Alloc = dhat::Alloc;

    #[test]
    fn test_tensor_memory_allocations() {
        let _profiler = dhat::Profiler::builder().testing().build();

        //  let nd_tensor = ndarray_base::Tensor1{
        //     inner: burn::prelude::Tensor::<NdArray,1>::from_floats([1.0], &crate::nn::NDARRAYDEVICE)
        //  };
        let tensor: burn::prelude::Tensor<burn::backend::NdArray, 1> =
            burn::prelude::Tensor::<NdArray, 1>::from_floats([1.0], &crate::nn::NDARRAYDEVICE);
        let stats = dhat::HeapStats::get();
        // dhat::assert_eq!(nd_tensor.inner, tensor);
        println!("Number of blocks {:#?}", stats.curr_blocks);
        println!("Number of bytes {:#?}", stats.curr_bytes);
    }
}
