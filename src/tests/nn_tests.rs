#![allow(unused)]

#[cfg(test)]
mod nn_test {
    use dhat;
    use crate::tensor::wgpu_base;
    use crate::tensor::ndarray_base;
    
    #[global_allocator]
    static ALLOC : dhat::Alloc = dhat::Alloc;
    
    #[test]
    fn test_tensor_memory_allocations(){
        let _profiler = dhat::Profiler::builder().testing().build();

        // let nd_tensor = ndarray_base::TensorPy::new();
        let stats = dhat::HeapStats::get();
        println!("Number of blocks {:#?}", stats.curr_blocks);
        println!("Number of bytes {:#?}", stats.curr_bytes);
    }
}