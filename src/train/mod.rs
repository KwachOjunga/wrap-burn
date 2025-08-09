use pyo3::prelude::*;

mod base;

#[pymodule]
pub mod wgpu_train {
    use super::*;

    #[pymodule_export]
    use super::base::wgpu::ClassificationOutputPy;
    #[pymodule_export]
    use super::base::wgpu::FileApplicationLoggerInstallerPy;
    #[pymodule_export]
    use super::base::wgpu::LearnerSummaryPy;
    #[pymodule_export]
    use super::base::wgpu::MetricEarlyStoppingStrategyPy;
    #[pymodule_export]
    use super::base::wgpu::MetricEntryPy;
    #[pymodule_export]
    use super::base::wgpu::MetricSummaryPy;
    #[pymodule_export]
    use super::base::wgpu::MultiLabelClassificationOutputPy;
    #[pymodule_export]
    use super::base::wgpu::RegressionOutputPy;
    #[pymodule_export]
    use super::base::wgpu::StoppingConditionPy;
    #[pymodule_export]
    use super::base::wgpu::SummaryMetricsPy;
    #[pymodule_export]
    use super::base::wgpu::TrainingInterrupterPy;

    #[pymodule]
    pub mod checkpoint {

        #[pymodule_export]
        use super::base::wgpu::checkpoint::CheckPointError;
        #[pymodule_export]
        use super::base::wgpu::checkpoint::CheckPointingActionPy;
        #[pymodule_export]
        use super::base::wgpu::checkpoint::ComposedCheckpointingStrategyBuilderPy;
        #[pymodule_export]
        use super::base::wgpu::checkpoint::ComposedCheckpointingStrategyPy;
        #[pymodule_export]
        use super::base::wgpu::checkpoint::KeepLastNCheckpointsPy;
        #[pymodule_export]
        use super::base::wgpu::checkpoint::MetricCheckpointingStrategyPy;
    }

    #[pymodule]
    pub mod metric {

        use super::*;

        #[pymodule_export]
        use super::base::wgpu::metric::AccuracyInputPy;
        #[pymodule_export]
        use super::base::wgpu::metric::AccuracyMetricPy;
        #[pymodule_export]
        use super::base::wgpu::metric::AurocInputPy;
        #[pymodule_export]
        use super::base::wgpu::metric::AurocMetricPy;
        #[pymodule_export]
        use super::base::wgpu::metric::ClassReductionPy;
        #[pymodule_export]
        use super::base::wgpu::metric::ConfusionStatsInputPy;
        #[pymodule_export]
        use super::base::wgpu::metric::CpuMemoryPy;
        #[pymodule_export]
        use super::base::wgpu::metric::CpuTemperaturePy;
        #[pymodule_export]
        use super::base::wgpu::metric::CpuUsePy;
        #[pymodule_export]
        use super::base::wgpu::metric::FBetaScoreMetricPy;
        #[pymodule_export]
        use super::base::wgpu::metric::HammingScoreInputPy;
        #[pymodule_export]
        use super::base::wgpu::metric::HammingScorePy;
        #[pymodule_export]
        use super::base::wgpu::metric::IterationSpeedMetricPy;
        #[pymodule_export]
        use super::base::wgpu::metric::LearningRateMetricPy;
        #[pymodule_export]
        use super::base::wgpu::metric::LossInputPy;
        #[pymodule_export]
        use super::base::wgpu::metric::LossMetricPy;
        #[pymodule_export]
        use super::base::wgpu::metric::MetricMetadataPy;
        #[pymodule_export]
        use super::base::wgpu::metric::NumericEntryPy;
        #[pymodule_export]
        use super::base::wgpu::metric::PrecisionMetricPy;
        #[pymodule_export]
        use super::base::wgpu::metric::RecallMetricPy;
        #[pymodule_export]
        use super::base::wgpu::metric::TopKAccuracyInputPy;
        #[pymodule_export]
        use super::base::wgpu::metric::TopKAccuracyMetricPy;

        #[pymodule]
        pub mod state {

            // use super::*;

            #[pymodule_export]
            use super::base::wgpu::metric::state::FormatOptionsPy;
            #[pymodule_export]
            use super::base::wgpu::metric::state::NumerMetricStatePy;
        }

        #[pymodule]
        pub mod store {
            // use super::*;

            #[pymodule_export]
            use super::base::wgpu::metric::store::AggregatePy;
            #[pymodule_export]
            use super::base::wgpu::metric::store::DirectionPy;
            #[pymodule_export]
            use super::base::wgpu::metric::store::EventPy;
            #[pymodule_export]
            use super::base::wgpu::metric::store::EventStoreClientPy;
            #[pymodule_export]
            use super::base::wgpu::metric::store::MetricsUpdatePy;
            #[pymodule_export]
            use super::base::wgpu::metric::store::SplitPy;
        }
    }

    #[pymodule]
    pub mod renderer {
        // use super::*;

        #[pymodule_export]
        use super::base::wgpu::renderer::MetricStatePy;
        #[pymodule_export]
        use super::base::wgpu::renderer::TrainingProgressPy;
        #[pymodule_export]
        use super::base::wgpu::renderer::TuiMetricsRendererPy;
    }
}

#[pymodule]
pub mod ndarray_train {
    use super::*;

    #[pymodule_export]
    use super::base::ndarray::ClassificationOutputPy;
    #[pymodule_export]
    use super::base::ndarray::FileApplicationLoggerInstallerPy;
    #[pymodule_export]
    use super::base::ndarray::LearnerSummaryPy;
    #[pymodule_export]
    use super::base::ndarray::MetricEarlyStoppingStrategyPy;
    #[pymodule_export]
    use super::base::ndarray::MetricEntryPy;
    #[pymodule_export]
    use super::base::ndarray::MetricSummaryPy;
    #[pymodule_export]
    use super::base::ndarray::MultiLabelClassificationOutputPy;
    #[pymodule_export]
    use super::base::ndarray::RegressionOutputPy;
    #[pymodule_export]
    use super::base::ndarray::StoppingConditionPy;
    #[pymodule_export]
    use super::base::ndarray::SummaryMetricsPy;
    #[pymodule_export]
    use super::base::ndarray::TrainingInterrupterPy;

    #[pymodule]
    pub mod checkpoint {

        #[pymodule_export]
        use super::base::ndarray::checkpoint::CheckPointError;
        #[pymodule_export]
        use super::base::ndarray::checkpoint::CheckPointingActionPy;
        #[pymodule_export]
        use super::base::ndarray::checkpoint::ComposedCheckpointingStrategyBuilderPy;
        #[pymodule_export]
        use super::base::ndarray::checkpoint::ComposedCheckpointingStrategyPy;
        #[pymodule_export]
        use super::base::ndarray::checkpoint::KeepLastNCheckpointsPy;
        #[pymodule_export]
        use super::base::ndarray::checkpoint::MetricCheckpointingStrategyPy;
    }

    #[pymodule]
    pub mod metric {

        use super::*;

        #[pymodule_export]
        use super::base::ndarray::metric::AccuracyInputPy;
        #[pymodule_export]
        use super::base::ndarray::metric::AccuracyMetricPy;
        #[pymodule_export]
        use super::base::ndarray::metric::AurocInputPy;
        #[pymodule_export]
        use super::base::ndarray::metric::AurocMetricPy;
        #[pymodule_export]
        use super::base::ndarray::metric::ClassReductionPy;
        #[pymodule_export]
        use super::base::ndarray::metric::ConfusionStatsInputPy;
        #[pymodule_export]
        use super::base::ndarray::metric::CpuMemoryPy;
        #[pymodule_export]
        use super::base::ndarray::metric::CpuTemperaturePy;
        #[pymodule_export]
        use super::base::ndarray::metric::CpuUsePy;
        #[pymodule_export]
        use super::base::ndarray::metric::FBetaScoreMetricPy;
        #[pymodule_export]
        use super::base::ndarray::metric::HammingScoreInputPy;
        #[pymodule_export]
        use super::base::ndarray::metric::HammingScorePy;
        #[pymodule_export]
        use super::base::ndarray::metric::IterationSpeedMetricPy;
        #[pymodule_export]
        use super::base::ndarray::metric::LearningRateMetricPy;
        #[pymodule_export]
        use super::base::ndarray::metric::LossInputPy;
        #[pymodule_export]
        use super::base::ndarray::metric::LossMetricPy;
        #[pymodule_export]
        use super::base::ndarray::metric::MetricMetadataPy;
        #[pymodule_export]
        use super::base::ndarray::metric::NumericEntryPy;
        #[pymodule_export]
        use super::base::ndarray::metric::PrecisionMetricPy;
        #[pymodule_export]
        use super::base::ndarray::metric::RecallMetricPy;
        #[pymodule_export]
        use super::base::ndarray::metric::TopKAccuracyInputPy;
        #[pymodule_export]
        use super::base::ndarray::metric::TopKAccuracyMetricPy;

        #[pymodule]
        pub mod state {

            // use super::*;

            #[pymodule_export]
            use super::base::ndarray::metric::state::FormatOptionsPy;
            #[pymodule_export]
            use super::base::ndarray::metric::state::NumerMetricStatePy;
        }

        #[pymodule]
        pub mod store {
            // use super::*;

            #[pymodule_export]
            use super::base::ndarray::metric::store::AggregatePy;
            #[pymodule_export]
            use super::base::ndarray::metric::store::DirectionPy;
            #[pymodule_export]
            use super::base::ndarray::metric::store::EventPy;
            #[pymodule_export]
            use super::base::ndarray::metric::store::EventStoreClientPy;
            #[pymodule_export]
            use super::base::ndarray::metric::store::MetricsUpdatePy;
            #[pymodule_export]
            use super::base::ndarray::metric::store::SplitPy;
        }
    }

    #[pymodule]
    pub mod renderer {
        // use super::*;

        #[pymodule_export]
        use super::base::ndarray::renderer::MetricStatePy;
        #[pymodule_export]
        use super::base::ndarray::renderer::TrainingProgressPy;
        #[pymodule_export]
        use super::base::ndarray::renderer::TuiMetricsRendererPy;
    }
}
