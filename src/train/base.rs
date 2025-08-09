#![allow(unused)]

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

    // find a way to implement learner, LearnerBuilder, MultiDevicesTrainStep, TrainEpoch.
    implement_wgpu_interface!(ClassificationOutputPy, ClassificationOutput);
    implement_wgpu_interface!(
        MultiLabelClassificationOutputPy,
        MultiLabelClassificationOutput
    );
    implement_wgpu_interface!(RegressionOutputPy, RegressionOutput);

    for_normal_struct_enums!(
        FileApplicationLoggerInstallerPy,
        FileApplicationLoggerInstaller
    );
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
        // implement_wgpu_interface!();

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
        implement_wgpu_interface!(AccuracyMetricPy, AccuracyMetric);
        implement_wgpu_interface!(AurocInputPy, AurocInput);
        implement_wgpu_interface!(AurocMetricPy, AurocMetric);
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

#[cfg(feature = "ndarray")]
pub mod ndarray {
    use super::*;

    // find a way to implement learner, LearnerBuilder, MultiDevicesTrainStep, TrainEpoch.
    implement_ndarray_interface!(ClassificationOutputPy, ClassificationOutput);
    implement_ndarray_interface!(
        MultiLabelClassificationOutputPy,
        MultiLabelClassificationOutput
    );
    implement_ndarray_interface!(RegressionOutputPy, RegressionOutput);

    for_normal_struct_enums!(
        FileApplicationLoggerInstallerPy,
        FileApplicationLoggerInstaller
    );
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
