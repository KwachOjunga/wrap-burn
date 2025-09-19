use pyo3::prelude::*;

mod base;

#[cfg(feature = "wgpu")]
#[pymodule]
pub mod wgpu_train {
    use super::*;

    #[pymodule]
    pub mod train {
        use super::*;

        #[pymodule_export]
        use super::base::wg_train::ClassificationOutputPy;
        #[pymodule_export]
        use super::base::wg_train::FileApplicationLoggerInstallerPy;
        #[pymodule_export]
        use super::base::wg_train::LearnerSummaryPy;
        #[pymodule_export]
        use super::base::wg_train::MetricEarlyStoppingStrategyPy;
        #[pymodule_export]
        use super::base::wg_train::MetricEntryPy;
        #[pymodule_export]
        use super::base::wg_train::MetricSummaryPy;
        #[pymodule_export]
        use super::base::wg_train::MultiLabelClassificationOutputPy;
        #[pymodule_export]
        use super::base::wg_train::RegressionOutputPy;
        #[pymodule_export]
        use super::base::wg_train::StoppingConditionPy;
        #[pymodule_export]
        use super::base::wg_train::SummaryMetricsPy;
        #[pymodule_export]
        use super::base::wg_train::TrainingInterrupterPy;

        #[pymodule]
        pub mod checkpoint {

            use super::*;

            #[pymodule_export]
            use super::base::wg_train::checkpoint::CheckPointError;
            #[pymodule_export]
            use super::base::wg_train::checkpoint::CheckPointingActionPy;
            #[pymodule_export]
            use super::base::wg_train::checkpoint::ComposedCheckpointingStrategyBuilderPy;
            #[pymodule_export]
            use super::base::wg_train::checkpoint::ComposedCheckpointingStrategyPy;
            #[pymodule_export]
            use super::base::wg_train::checkpoint::KeepLastNCheckpointsPy;
            #[pymodule_export]
            use super::base::wg_train::checkpoint::MetricCheckpointingStrategyPy;
        }

        #[pymodule]
        pub mod metric {

            use super::*;

            #[pymodule_export]
            use super::base::wg_train::metric::AccuracyInputPy;
            #[pymodule_export]
            use super::base::wg_train::metric::AccuracyMetricPy;
            #[pymodule_export]
            use super::base::wg_train::metric::AurocInputPy;
            #[pymodule_export]
            use super::base::wg_train::metric::AurocMetricPy;
            #[pymodule_export]
            use super::base::wg_train::metric::ClassReductionPy;
            #[pymodule_export]
            use super::base::wg_train::metric::ConfusionStatsInputPy;
            #[pymodule_export]
            use super::base::wg_train::metric::CpuMemoryPy;
            #[pymodule_export]
            use super::base::wg_train::metric::CpuTemperaturePy;
            #[pymodule_export]
            use super::base::wg_train::metric::CpuUsePy;
            #[pymodule_export]
            use super::base::wg_train::metric::FBetaScoreMetricPy;
            #[pymodule_export]
            use super::base::wg_train::metric::HammingScoreInputPy;
            #[pymodule_export]
            use super::base::wg_train::metric::HammingScorePy;
            #[pymodule_export]
            use super::base::wg_train::metric::IterationSpeedMetricPy;
            #[pymodule_export]
            use super::base::wg_train::metric::LearningRateMetricPy;
            #[pymodule_export]
            use super::base::wg_train::metric::LossInputPy;
            #[pymodule_export]
            use super::base::wg_train::metric::LossMetricPy;
            #[pymodule_export]
            use super::base::wg_train::metric::MetricMetadataPy;
            #[pymodule_export]
            use super::base::wg_train::metric::NumericEntryPy;
            #[pymodule_export]
            use super::base::wg_train::metric::PrecisionMetricPy;
            #[pymodule_export]
            use super::base::wg_train::metric::RecallMetricPy;
            #[pymodule_export]
            use super::base::wg_train::metric::TopKAccuracyInputPy;
            #[pymodule_export]
            use super::base::wg_train::metric::TopKAccuracyMetricPy;

            #[pymodule]
            pub mod state {

                use super::*;

                #[pymodule_export]
                use super::base::wg_train::metric::state::FormatOptionsPy;
                #[pymodule_export]
                use super::base::wg_train::metric::state::NumerMetricStatePy;
            }

            #[pymodule]
            pub mod store {
                use super::*;

                #[pymodule_export]
                use super::base::wg_train::metric::store::AggregatePy;
                #[pymodule_export]
                use super::base::wg_train::metric::store::DirectionPy;
                #[pymodule_export]
                use super::base::wg_train::metric::store::EventPy;
                #[pymodule_export]
                use super::base::wg_train::metric::store::EventStoreClientPy;
                #[pymodule_export]
                use super::base::wg_train::metric::store::MetricsUpdatePy;
                #[pymodule_export]
                use super::base::wg_train::metric::store::SplitPy;
            }
        }

        #[pymodule]
        pub mod renderer {
            use super::*;

            #[pymodule_export]
            use super::base::wg_train::renderer::MetricStatePy;
            #[pymodule_export]
            use super::base::wg_train::renderer::TrainingProgressPy;
            #[pymodule_export]
            use super::base::wg_train::renderer::TuiMetricsRendererPy;
        }
    }
}
