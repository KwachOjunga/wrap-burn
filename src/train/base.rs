#![allow(unused)]

use crate::tensor::tensor_error::TensorError;
use crate::{
    for_normal_struct_enums, implement_ndarray_interface, implement_send_and_sync,
    implement_wgpu_interface,
};
use burn::prelude::*;
use burn::train::*;
use pyo3::prelude::*;

#[cfg(feature = "wgpu")]
pub mod wgpu {
    use super::*;
    use crate::tensor::wgpu_base::TensorPy;

    // find a way to implement learner, LearnerBuilder, MultiDevicesTrainStep, TrainEpoch.
    implement_wgpu_interface!(ClassificationOutputPy, ClassificationOutput);

    impl From<ClassificationOutput<Wgpu>> for ClassificationOutputPy {
        fn from(other: ClassificationOutput<Wgpu>) -> Self {
            Self { inner: other }
        }
    }
    #[pymethods]
    impl ClassificationOutputPy {
        #[new]
        fn new(
            loss: TensorPy,    /*Tensor<B, 1>,*/
            output: TensorPy,  /*Tensor<B, 2>*/
            targets: TensorPy, /*Tensor<B, 1, Int>*/
        ) -> Self {
            let loss = match loss {
                TensorPy::TensorOne(val) => Ok(val.inner),
                _ => Err(TensorError::WrongDimensions),
            };
            let output = match output {
                TensorPy::TensorTwo(val) => Ok(val.inner),
                _ => Err(TensorError::WrongDimensions),
            };
            let targets = match targets {
                TensorPy::TensorOneInt(val) => Ok(val.inner),
                _ => Err(TensorError::WrongDimensions),
            };
            ClassificationOutput::new(loss.unwrap(), output.unwrap(), targets.unwrap()).into()
        }
    }

    implement_wgpu_interface!(
        MultiLabelClassificationOutputPy,
        MultiLabelClassificationOutput
    );

    #[pymethods]
    impl MultiLabelClassificationOutputPy {
        #[new]
        fn new(
            loss: TensorPy,    /*Tensor<B, 1>,*/
            output: TensorPy,  /*Tensor<B, 2>*/
            targets: TensorPy, /*Tensor<B, 1, Int>*/
        ) -> Self {
            let loss = match loss {
                TensorPy::TensorOne(val) => Ok(val.inner),
                _ => Err(TensorError::WrongDimensions),
            };
            let output = match output {
                TensorPy::TensorTwo(val) => Ok(val.inner),
                _ => Err(TensorError::WrongDimensions),
            };
            let targets = match targets {
                TensorPy::TensorTwoInt(val) => Ok(val.inner),
                _ => Err(TensorError::WrongDimensions),
            };
            MultiLabelClassificationOutput::new(loss.unwrap(), output.unwrap(), targets.unwrap())
                .into()
        }
    }

    impl From<MultiLabelClassificationOutput<Wgpu>> for MultiLabelClassificationOutputPy {
        fn from(other: MultiLabelClassificationOutput<Wgpu>) -> Self {
            Self { inner: other }
        }
    }

    implement_wgpu_interface!(RegressionOutputPy, RegressionOutput);

    impl From<RegressionOutput<Wgpu>> for RegressionOutputPy {
        fn from(other: RegressionOutput<Wgpu>) -> Self {
            Self { inner: other }
        }
    }

    #[pymethods]
    impl RegressionOutputPy {
        #[new]
        fn new(
            loss: TensorPy,    /*Tensor<B, 1>,*/
            output: TensorPy,  /*Tensor<B, 2>*/
            targets: TensorPy, /*Tensor<B, 1, Int>*/
        ) -> Self {
            let loss = match loss {
                TensorPy::TensorOne(val) => Ok(val.inner),
                _ => Err(TensorError::WrongDimensions),
            };
            let output = match output {
                TensorPy::TensorTwo(val) => Ok(val.inner),
                _ => Err(TensorError::WrongDimensions),
            };
            let targets = match targets {
                TensorPy::TensorTwo(val) => Ok(val.inner),
                _ => Err(TensorError::WrongDimensions),
            };
            RegressionOutput::new(loss.unwrap(), output.unwrap(), targets.unwrap()).into()
        }
    }

    for_normal_struct_enums!(
        FileApplicationLoggerInstallerPy,
        FileApplicationLoggerInstaller
    );

    #[pymethods]
    impl FileApplicationLoggerInstallerPy {
        #[new]
        fn new(path: &str) -> Self {
            FileApplicationLoggerInstaller::new(path).into()
        }

        fn install(&self) -> PyResult<()> {
            Ok(self.0.install().unwrap())
        }
    }

    for_normal_struct_enums!(LearnerSummaryPy, LearnerSummary);
    for_normal_struct_enums!(MetricEarlyStoppingStrategyPy, MetricEarlyStoppingStrategy);
    for_normal_struct_enums!(MetricEntryPy, MetricEntry);
    for_normal_struct_enums!(MetricSummaryPy, MetricSummary);
    for_normal_struct_enums!(SummaryMetricsPy, SummaryMetrics);
    for_normal_struct_enums!(TrainingInterrupterPy, TrainingInterrupter);
    for_normal_struct_enums!(StoppingConditionPy, StoppingCondition);

    pub mod checkpoint {
        use super::*;
        use crate::record::PyRecorderError;
        use burn::train::checkpoint::*;
        use pyo3::exceptions::PyValueError;
        // [`TODO`] FileCheckpointer
        // implement_wgpu_interface!(PyAsyncCheckPointer,AsyncCheckpointer);

        for_normal_struct_enums!(
            ComposedCheckpointingStrategyPy,
            ComposedCheckpointingStrategy
        );
        for_normal_struct_enums!(
            ComposedCheckpointingStrategyBuilderPy,
            ComposedCheckpointingStrategyBuilder
        );
        for_normal_struct_enums!(KeepLastNCheckpointsPy, KeepLastNCheckpoints);
        for_normal_struct_enums!(MetricCheckpointingStrategyPy, MetricCheckpointingStrategy);
        for_normal_struct_enums!(CheckPointingActionPy, CheckpointingAction);
        implement_send_and_sync!(ComposedCheckpointingStrategyPy);
        implement_send_and_sync!(ComposedCheckpointingStrategyBuilderPy);

        // Errors
        #[pyclass]
        pub struct CheckPointError {
            pub inner: CheckpointerError,
        }

        impl From<CheckPointError> for PyErr {
            fn from(error: CheckPointError) -> Self {
                match error.inner {
                    CheckpointerError::IOError(err) => PyValueError::new_err(err),
                    CheckpointerError::RecorderError(err) => {
                        PyValueError::new_err::<PyRecorderError>(err.into())
                    }
                    CheckpointerError::Unknown(_) => {
                        PyValueError::new_err("unknown checkpoint error")
                    }
                }
            }
        }

        impl From<CheckpointerError> for CheckPointError {
            fn from(checkpointer_error: CheckpointerError) -> Self {
                Self {
                    inner: checkpointer_error,
                }
            }
        }
    }

    pub mod metric {
        pub(crate) use burn::train::metric::*;

        use super::*;

        implement_wgpu_interface!(AccuracyInputPy, AccuracyInput);

        impl From<AccuracyInput<Wgpu>> for AccuracyInputPy {
            fn from(other: AccuracyInput<Wgpu>) -> Self {
                Self { inner: other }
            }
        }

        #[pymethods]
        impl AccuracyInputPy {
            #[new]
            fn new(
                outputs: TensorPy, /*Tensor<B, 2>*/
                targets: TensorPy, /*Tensor<B, 1, Int>*/
            ) -> Self {
                let out = match outputs {
                    TensorPy::TensorTwo(val) => Ok(val.inner),
                    _ => Err(TensorError::WrongDimensions),
                };
                let target = match targets {
                    TensorPy::TensorOneInt(val) => Ok(val.inner),
                    _ => Err(TensorError::WrongDimensions),
                };
                AccuracyInput::new(out.unwrap(), target.unwrap()).into()
            }
        }

        implement_wgpu_interface!(AccuracyMetricPy, AccuracyMetric);

        impl From<AccuracyMetric<Wgpu>> for AccuracyMetricPy {
            fn from(other: AccuracyMetric<Wgpu>) -> Self {
                Self { inner: other }
            }
        }

        #[pymethods]
        impl AccuracyMetricPy {
            #[new]
            #[pyo3(signature = (pad_token = None))]
            fn new(pad_token: Option<usize>) -> Self {
                match pad_token {
                    Some(val) => AccuracyMetric::new().with_pad_token(val).into(),
                    None => AccuracyMetric::new().into(),
                }
            }
        }
        implement_wgpu_interface!(AurocInputPy, AurocInput);

        impl From<AurocInput<Wgpu>> for AurocInputPy {
            fn from(other: AurocInput<Wgpu>) -> Self {
                Self { inner: other }
            }
        }

        #[pymethods]
        impl AurocInputPy {
            #[new]
            fn new(output: TensorPy, targets: TensorPy) -> Self {
                let out = match output {
                    TensorPy::TensorTwo(val) => Ok(val.inner),
                    _ => Err(TensorError::WrongDimensions),
                };
                let targets = match targets {
                    TensorPy::TensorOneInt(val) => Ok(val.inner),
                    _ => Err(TensorError::WrongDimensions),
                };

                AurocInput::new(out.unwrap(), targets.unwrap()).into()
            }
        }

        implement_wgpu_interface!(AurocMetricPy, AurocMetric);

        impl From<AurocMetric<Wgpu>> for AurocMetricPy {
            fn from(other: AurocMetric<Wgpu>) -> Self {
                Self { inner: other }
            }
        }

        #[pymethods]
        impl AurocMetricPy {
            #[new]
            fn new() -> Self {
                AurocMetric::new().into()
            }
        }

        implement_wgpu_interface!(ConfusionStatsInputPy, ConfusionStatsInput);
        implement_wgpu_interface!(FBetaScoreMetricPy, FBetaScoreMetric);
        implement_wgpu_interface!(HammingScorePy, HammingScore);
        implement_wgpu_interface!(HammingScoreInputPy, HammingScoreInput);
        implement_wgpu_interface!(LossInputPy, LossInput);
        implement_wgpu_interface!(LossMetricPy, LossMetric);
        implement_wgpu_interface!(PrecisionMetricPy, PrecisionMetric);
        implement_wgpu_interface!(RecallMetricPy, RecallMetric);
        implement_wgpu_interface!(TopKAccuracyInputPy, TopKAccuracyInput);
        implement_wgpu_interface!(TopKAccuracyMetricPy, TopKAccuracyMetric);

        for_normal_struct_enums!(IterationSpeedMetricPy, IterationSpeedMetric);
        for_normal_struct_enums!(LearningRateMetricPy, LearningRateMetric);

        for_normal_struct_enums!(CpuMemoryPy, CpuMemory);
        for_normal_struct_enums!(CpuTemperaturePy, CpuTemperature);
        for_normal_struct_enums!(CpuUsePy, CpuUse);
        // for_normal_struct_enums!(PyMetricEntry,MetricEntry); --re-exported

        // [TODO: ] @kwach This type uses private structures.
        // It may require the position of burn in its ecosystem.
        for_normal_struct_enums!(MetricMetadataPy, MetricMetadata);

        // #[pymethods]
        // impl MetricMetadataPy {

        // }
        for_normal_struct_enums!(ClassReductionPy, ClassReduction);
        for_normal_struct_enums!(NumericEntryPy, NumericEntry);

        pub mod state {
            use super::*;
            use burn::train::metric::state::*;

            for_normal_struct_enums!(FormatOptionsPy, FormatOptions);
            for_normal_struct_enums!(NumerMetricStatePy, NumericMetricState);
        }

        pub mod store {
            use super::*;
            use burn::train::metric::store::*;

            for_normal_struct_enums!(EventStoreClientPy, EventStoreClient);

            for_normal_struct_enums!(MetricsUpdatePy, MetricsUpdate);

            for_normal_struct_enums!(AggregatePy, Aggregate);
            for_normal_struct_enums!(DirectionPy, Direction);

            for_normal_struct_enums!(EventPy, Event);

            for_normal_struct_enums!(SplitPy, Split);
        }
    }

    pub mod renderer {
        use super::*;
        use burn::train::renderer::{tui::*, *};

        for_normal_struct_enums!(MetricStatePy, MetricState);
        for_normal_struct_enums!(TrainingProgressPy, TrainingProgress);
        for_normal_struct_enums!(TuiMetricsRendererPy, TuiMetricsRenderer);
    }
}

#[cfg(feature = "ndarray")]
pub mod ndarray {
    use super::*;
    use crate::tensor::ndarray_base::TensorPy;

    // find a way to implement learner, LearnerBuilder, MultiDevicesTrainStep, TrainEpoch.
    implement_ndarray_interface!(ClassificationOutputPy, ClassificationOutput);

    impl From<ClassificationOutput<NdArray>> for ClassificationOutputPy {
        fn from(other: ClassificationOutput<NdArray>) -> Self {
            Self { inner: other }
        }
    }
    #[pymethods]
    impl ClassificationOutputPy {
        #[new]
        fn new(
            loss: TensorPy,    /*Tensor<B, 1>,*/
            output: TensorPy,  /*Tensor<B, 2>*/
            targets: TensorPy, /*Tensor<B, 1, Int>*/
        ) -> Self {
            let loss = match loss {
                TensorPy::TensorOne(val) => Ok(val.inner),
                _ => Err(TensorError::WrongDimensions),
            };
            let output = match output {
                TensorPy::TensorTwo(val) => Ok(val.inner),
                _ => Err(TensorError::WrongDimensions),
            };
            let targets = match targets {
                TensorPy::TensorOneInt(val) => Ok(val.inner),
                _ => Err(TensorError::WrongDimensions),
            };
            ClassificationOutput::new(loss.unwrap(), output.unwrap(), targets.unwrap()).into()
        }
    }

    implement_ndarray_interface!(
        MultiLabelClassificationOutputPy,
        MultiLabelClassificationOutput
    );

    impl From<MultiLabelClassificationOutput<NdArray>> for MultiLabelClassificationOutputPy {
        fn from(other: MultiLabelClassificationOutput<NdArray>) -> Self {
            Self { inner: other }
        }
    }

    #[pymethods]
    impl MultiLabelClassificationOutputPy {
        #[new]
        fn new(
            loss: TensorPy,    /*Tensor<B, 1>,*/
            output: TensorPy,  /*Tensor<B, 2>*/
            targets: TensorPy, /*Tensor<B, 1, Int>*/
        ) -> Self {
            let loss = match loss {
                TensorPy::TensorOne(val) => Ok(val.inner),
                _ => Err(TensorError::WrongDimensions),
            };
            let output = match output {
                TensorPy::TensorTwo(val) => Ok(val.inner),
                _ => Err(TensorError::WrongDimensions),
            };
            let targets = match targets {
                TensorPy::TensorTwoInt(val) => Ok(val.inner),
                _ => Err(TensorError::WrongDimensions),
            };
            MultiLabelClassificationOutput::new(loss.unwrap(), output.unwrap(), targets.unwrap())
                .into()
        }
    }

    implement_ndarray_interface!(RegressionOutputPy, RegressionOutput);

    impl From<RegressionOutput<NdArray>> for RegressionOutputPy {
        fn from(other: RegressionOutput<NdArray>) -> Self {
            Self { inner: other }
        }
    }

    #[pymethods]
    impl RegressionOutputPy {
        #[new]
        fn new(
            loss: TensorPy,    /*Tensor<B, 1>,*/
            output: TensorPy,  /*Tensor<B, 2>*/
            targets: TensorPy, /*Tensor<B, 1, Int>*/
        ) -> Self {
            let loss = match loss {
                TensorPy::TensorOne(val) => Ok(val.inner),
                _ => Err(TensorError::WrongDimensions),
            };
            let output = match output {
                TensorPy::TensorTwo(val) => Ok(val.inner),
                _ => Err(TensorError::WrongDimensions),
            };
            let targets = match targets {
                TensorPy::TensorTwo(val) => Ok(val.inner),
                _ => Err(TensorError::WrongDimensions),
            };
            RegressionOutput::new(loss.unwrap(), output.unwrap(), targets.unwrap()).into()
        }
    }

    for_normal_struct_enums!(
        FileApplicationLoggerInstallerPy,
        FileApplicationLoggerInstaller
    );

    #[pymethods]
    impl FileApplicationLoggerInstallerPy {
        #[new]
        fn new(path: &str) -> Self {
            FileApplicationLoggerInstaller::new(path).into()
        }

        fn install(&self) -> PyResult<()> {
            Ok(self.0.install().unwrap())
        }
    }
    for_normal_struct_enums!(LearnerSummaryPy, LearnerSummary);
    for_normal_struct_enums!(MetricEarlyStoppingStrategyPy, MetricEarlyStoppingStrategy);
    for_normal_struct_enums!(MetricEntryPy, MetricEntry);
    for_normal_struct_enums!(MetricSummaryPy, MetricSummary);
    for_normal_struct_enums!(SummaryMetricsPy, SummaryMetrics);
    for_normal_struct_enums!(TrainingInterrupterPy, TrainingInterrupter);
    for_normal_struct_enums!(StoppingConditionPy, StoppingCondition);

    pub mod checkpoint {
        use super::*;
        use crate::record::PyRecorderError;
        use burn::train::checkpoint::*;
        use pyo3::exceptions::PyValueError;
        // [`TODO`] FileCheckpointer
        // implement_ndarray_interface!(PyAsyncCheckPointer,AsyncCheckpointer);
        // implement_ndarray_interface!();

        for_normal_struct_enums!(
            ComposedCheckpointingStrategyPy,
            ComposedCheckpointingStrategy
        );
        for_normal_struct_enums!(
            ComposedCheckpointingStrategyBuilderPy,
            ComposedCheckpointingStrategyBuilder
        );
        for_normal_struct_enums!(KeepLastNCheckpointsPy, KeepLastNCheckpoints);
        for_normal_struct_enums!(MetricCheckpointingStrategyPy, MetricCheckpointingStrategy);
        for_normal_struct_enums!(CheckPointingActionPy, CheckpointingAction);
        implement_send_and_sync!(ComposedCheckpointingStrategyPy);
        implement_send_and_sync!(ComposedCheckpointingStrategyBuilderPy);

        // Errors
        #[pyclass]
        pub struct CheckPointError {
            pub inner: CheckpointerError,
        }

        impl From<CheckPointError> for PyErr {
            fn from(error: CheckPointError) -> Self {
                match error.inner {
                    CheckpointerError::IOError(err) => PyValueError::new_err(err),
                    CheckpointerError::RecorderError(err) => {
                        PyValueError::new_err::<PyRecorderError>(err.into())
                    }
                    CheckpointerError::Unknown(_) => {
                        PyValueError::new_err("unknown checkpoint error")
                    }
                }
            }
        }

        impl From<CheckpointerError> for CheckPointError {
            fn from(checkpointer_error: CheckpointerError) -> Self {
                Self {
                    inner: checkpointer_error,
                }
            }
        }
    }

    pub mod metric {
        pub(crate) use burn::train::metric::*;

        use super::*;

        implement_ndarray_interface!(AccuracyInputPy, AccuracyInput);
        implement_ndarray_interface!(AccuracyMetricPy, AccuracyMetric);
        implement_ndarray_interface!(AurocInputPy, AurocInput);
        implement_ndarray_interface!(AurocMetricPy, AurocMetric);
        implement_ndarray_interface!(ConfusionStatsInputPy, ConfusionStatsInput);
        implement_ndarray_interface!(FBetaScoreMetricPy, FBetaScoreMetric);
        implement_ndarray_interface!(HammingScorePy, HammingScore);
        implement_ndarray_interface!(HammingScoreInputPy, HammingScoreInput);
        implement_ndarray_interface!(LossInputPy, LossInput);
        implement_ndarray_interface!(LossMetricPy, LossMetric);
        implement_ndarray_interface!(PrecisionMetricPy, PrecisionMetric);
        implement_ndarray_interface!(RecallMetricPy, RecallMetric);
        implement_ndarray_interface!(TopKAccuracyInputPy, TopKAccuracyInput);
        implement_ndarray_interface!(TopKAccuracyMetricPy, TopKAccuracyMetric);

        for_normal_struct_enums!(IterationSpeedMetricPy, IterationSpeedMetric);
        for_normal_struct_enums!(LearningRateMetricPy, LearningRateMetric);

        for_normal_struct_enums!(CpuMemoryPy, CpuMemory);
        for_normal_struct_enums!(CpuTemperaturePy, CpuTemperature);
        for_normal_struct_enums!(CpuUsePy, CpuUse);
        // for_normal_struct_enums!(PyMetricEntry,MetricEntry); --re-exported
        for_normal_struct_enums!(MetricMetadataPy, MetricMetadata);
        for_normal_struct_enums!(ClassReductionPy, ClassReduction);
        for_normal_struct_enums!(NumericEntryPy, NumericEntry);

        pub mod state {
            use super::*;
            use burn::train::metric::state::*;

            for_normal_struct_enums!(FormatOptionsPy, FormatOptions);
            for_normal_struct_enums!(NumerMetricStatePy, NumericMetricState);
        }

        pub mod store {
            use super::*;
            use burn::train::metric::store::*;

            for_normal_struct_enums!(EventStoreClientPy, EventStoreClient);
            for_normal_struct_enums!(MetricsUpdatePy, MetricsUpdate);
            for_normal_struct_enums!(AggregatePy, Aggregate);
            for_normal_struct_enums!(DirectionPy, Direction);
            for_normal_struct_enums!(EventPy, Event);
            for_normal_struct_enums!(SplitPy, Split);
        }
    }

    pub mod renderer {
        use super::*;
        use burn::train::renderer::{tui::*, *};

        for_normal_struct_enums!(MetricStatePy, MetricState);
        for_normal_struct_enums!(TrainingProgressPy, TrainingProgress);
        for_normal_struct_enums!(TuiMetricsRendererPy, TuiMetricsRenderer);
    }
}
