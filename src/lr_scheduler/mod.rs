use crate::for_normal_struct_enums;
use burn::lr_scheduler::constant::*;
use burn::lr_scheduler::cosine::*;
use burn::lr_scheduler::exponential::*;
use burn::lr_scheduler::linear::*;
use burn::lr_scheduler::noam::*;
use burn::lr_scheduler::step::*;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;

#[pymodule]
pub mod scheduler {
    use super::*;

    for_normal_struct_enums!(
        /// Learning rate scheduler that keeps the learning rate constant.
        ConstantLrPy,
        ConstantLr,
        "Learning rate scheduler that keeps the learning rate constant."
    );

    #[pymethods]
    impl ConstantLrPy {
        #[new]
        #[pyo3(signature = (val))]
        fn new(val: f64) -> Self {
            ConstantLr::new(val).into()
        }
    }

    for_normal_struct_enums!(
        /// Learning rate scheduler that decays the learning rate by a factor every specified number of steps.
        StepLrSchedulerPy,
        StepLrScheduler,
        "Learning rate scheduler that decays the learning rate by a factor every specified number of steps."
    );

    #[pymethods]
    impl StepLrSchedulerPy {
        #[new]
        #[pyo3(signature = (step_size, initial_lr, gamma = None))]
        fn new(step_size: f64, initial_lr: usize, gamma: Option<f64>) -> PyResult<Self> {
            let g = match gamma {
                Some(val) => val,
                None => 0.1,
            };
            let step = StepLrSchedulerConfig::new(step_size, initial_lr)
                .with_gamma(g)
                .init();
            match step {
                Ok(val) => Ok(val.into()),
                Err(err) => Err(PyErr::new::<PyException, _>(err)),
            }
        }
    }

    for_normal_struct_enums!(
        /// Learning rate scheduler that linearly increases the learning rate from an initial value to a final value over a specified number of steps.
        LinearLrSchedulerPy,
        LinearLrScheduler,
        "Learning rate scheduler that linearly increases the learning rate from an initial value to a final value over a specified number of steps."
    );

    #[pymethods]
    impl LinearLrSchedulerPy {
        #[new]
        #[pyo3(signature = (initial_lr, final_lr, num_iters))]
        fn new(initial_lr: f64, final_lr: f64, num_iters: usize) -> PyResult<Self> {
            let sched = LinearLrSchedulerConfig::new(initial_lr, final_lr, num_iters).init();
            match sched {
                Ok(val) => Ok(val.into()),
                Err(err) => Err(PyErr::new::<PyException, _>(err)),
            }
        }
    }

    for_normal_struct_enums!(
        /// Learning rate scheduler that adjusts the learning rate following a cosine decay pattern.
        CosineAnnealingLrSchedulerPy,
        CosineAnnealingLrScheduler,
        "Learning rate scheduler that adjusts the learning rate following a cosine decay pattern."
    );

    #[pymethods]
    impl CosineAnnealingLrSchedulerPy {
        #[new]
        #[pyo3(signature = (initial_lr, num_iters, min_lr = None))]
        fn new(initial_lr: f64, num_iters: usize, min_lr: Option<f64>) -> PyResult<Self> {
            let min_lr = match min_lr {
                Some(val) => val,
                None => 0.0,
            };
            let sched = CosineAnnealingLrSchedulerConfig::new(initial_lr, num_iters)
                .with_min_lr(min_lr)
                .init();
            match sched {
                Ok(val) => Ok(val.into()),
                Err(err) => Err(PyErr::new::<PyException, _>(err)),
            }
        }
    }

    for_normal_struct_enums!(
        /// Learning rate scheduler that exponentially decays the learning rate by a specified factor at each step.
        ExponentialLrSchedulerPy,
        ExponentialLrScheduler,
        "Learning rate scheduler that exponentially decays the learning rate by a specified factor at each step."
    );

    #[pymethods]
    impl ExponentialLrSchedulerPy {
        #[new]
        #[pyo3(signature = (initial_lr, gamma))]
        fn new(initial_lr: f64, gamma: f64) -> PyResult<Self> {
            let sched = ExponentialLrSchedulerConfig::new(initial_lr, gamma).init();
            match sched {
                Ok(val) => Ok(val.into()),
                Err(err) => Err(PyErr::new::<PyException, _>(err)),
            }
        }
    }

    for_normal_struct_enums!(
        /// Learning rate scheduler that implements the Noam learning rate schedule, commonly used in transformer models.
        NoamLrSchedulerPy,
        NoamLrScheduler,
        "Learning rate scheduler that implements the Noam learning rate schedule, commonly used in transformer models."
    );

    #[pymethods]
    impl NoamLrSchedulerPy {
        #[new]
        #[pyo3(signature = (init_lr, warmup_steps = None, model_size = None))]
        fn new(
            init_lr: f64,
            warmup_steps: Option<usize>,
            model_size: Option<usize>,
        ) -> PyResult<Self> {
            let warm_up = match warmup_steps {
                Some(val) => val,
                None => 4000,
            };
            let model_size = match model_size {
                Some(val) => val,
                None => 512,
            };
            let sched = NoamLrSchedulerConfig::new(init_lr)
                .with_warmup_steps(warm_up)
                .with_model_size(model_size)
                .init();
            match sched {
                Ok(val) => Ok(val.into()),
                Err(err) => Err(PyErr::new::<PyException, _>(err)),
            }
        }
    }
}
