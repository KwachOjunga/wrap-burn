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
    implement_wgpu_interface!(PyClassificationOutput, ClassificationOutput);
    implement_wgpu_interface!(
        PyMultiLabelClassificationOutput,
        MultiLabelClassificationOutput
    );
    implement_wgpu_interface!(PyRegressionOutput, RegressionOutput);

    for_normal_struct_enums!(
        PyFileApplicationLoggerInstaller,
        FileApplicationLoggerInstaller
    );
    for_normal_struct_enums!(PyLearnerSummary, LearnerSummary);
    for_normal_struct_enums!(PyMetricEarlyStoppingStrategy, MetricEarlyStoppingStrategy);
    for_normal_struct_enums!(PyMetricEntry, MetricEntry);
    for_normal_struct_enums!(PyMetricSummary, MetricSummary);
    for_normal_struct_enums!(PySummaryMetrics, SummaryMetrics);
    for_normal_struct_enums!(PyTrainingInterrupter, TrainingInterrupter);
    for_normal_struct_enums!(PyStoppingCondition, StoppingCondition);

    pub mod checkpoint {
        use super::*;
        use crate::record::PyRecorderError;
        use burn::train::checkpoint::*;
        use pyo3::exceptions::PyValueError;
        // [`TODO`] FileCheckpointer
        // implement_wgpu_interface!(PyAsyncCheckPointer,AsyncCheckpointer);
        // implement_wgpu_interface!();

        for_normal_struct_enums!(
            PyComposedCheckpointingStrategy,
            ComposedCheckpointingStrategy
        );
        for_normal_struct_enums!(
            PyComposedCheckpointingStrategyBuilder,
            ComposedCheckpointingStrategyBuilder
        );
        for_normal_struct_enums!(PyKeepLastNCheckpoints, KeepLastNCheckpoints);
        for_normal_struct_enums!(PyMetricCheckpointingStrategy, MetricCheckpointingStrategy);
        for_normal_struct_enums!(PyCheckPointingAction, CheckpointingAction);
        implement_send_and_sync!(PyComposedCheckpointingStrategy);
        implement_send_and_sync!(PyComposedCheckpointingStrategyBuilder);

        // Errors
        #[pyclass]
        struct CheckPointError {
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

        implement_wgpu_interface!(PyAccuracyInput, AccuracyInput);
        implement_wgpu_interface!(PyAccuracyMetric, AccuracyMetric);
        implement_wgpu_interface!(PyAurocInput, AurocInput);
        implement_wgpu_interface!(PyAurocMetric, AurocMetric);
        implement_wgpu_interface!(PyConfusionStatsInput, ConfusionStatsInput);
        implement_wgpu_interface!(PyFBetaScoreMetric, FBetaScoreMetric);
        implement_wgpu_interface!(PyHammingScore, HammingScore);
        implement_wgpu_interface!(PyHammingScoreInput, HammingScoreInput);
        implement_wgpu_interface!(PyLossInput, LossInput);
        implement_wgpu_interface!(PyLossMetric, LossMetric);
        implement_wgpu_interface!(PyPrecisionMetric, PrecisionMetric);
        implement_wgpu_interface!(PyRecallMetric, RecallMetric);
        implement_wgpu_interface!(PyTopKAccuracyInput, TopKAccuracyInput);
        implement_wgpu_interface!(PyTopKAccuracyMetric, TopKAccuracyMetric);

        for_normal_struct_enums!(PyIterationSpeedMetric, IterationSpeedMetric);
        for_normal_struct_enums!(PyLearningRateMetric, LearningRateMetric);

        for_normal_struct_enums!(PyCpuMemory, CpuMemory);
        for_normal_struct_enums!(PyCpuTemperature, CpuTemperature);
        for_normal_struct_enums!(PyCpuUse, CpuUse);
        // for_normal_struct_enums!(PyMetricEntry,MetricEntry); --re-exported
        for_normal_struct_enums!(PyMetricMetadata, MetricMetadata);
        for_normal_struct_enums!(PyClassReduction, ClassReduction);
        for_normal_struct_enums!(PyNumericEntry, NumericEntry);

        pub mod state {
            use super::*;
            use burn::train::metric::state::*;

            for_normal_struct_enums!(PyFormatOptions, FormatOptions);
            for_normal_struct_enums!(PyNumerMetricState, NumericMetricState);
        }

        pub mod store {
            use super::*;
            use burn::train::metric::store::*;

            for_normal_struct_enums!(PyEventStoreClient, EventStoreClient);
            for_normal_struct_enums!(PyMetricsUpdate, MetricsUpdate);
            for_normal_struct_enums!(PyAggregate, Aggregate);
            for_normal_struct_enums!(PyDirection, Direction);
            for_normal_struct_enums!(PyEvent, Event);
            for_normal_struct_enums!(PySplit, Split);
        }
    }

    pub mod renderer {
        use super::*;
        use burn::train::renderer::{tui::*, *};

        for_normal_struct_enums!(PyMetricState, MetricState);
        for_normal_struct_enums!(PyTrainingProgress, TrainingProgress);
        for_normal_struct_enums!(PyTuiMetricsRenderer, TuiMetricsRenderer);
    }
    // implement_send_and_sync!()
}

#[cfg(feature = "ndarray")]
pub mod ndarray {}
