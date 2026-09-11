//! Progressive Pipeline Factory
//!
//! Creates ISP-compliant progressive search pipelines based on engine type
//! and RL planner actions. This module serves as the integration point between
//! the RL query planner and the engine-specific progressive stages.

use std::sync::Arc;

use super::{ProgressiveSearchCoordinator, ProgressiveSearchStage};
use crate::compute::quantization::quantization_engine::UnifiedQuantizationEngine;
use proximadb_distance_kernel::engine::UnifiedDistanceCompute;

/// Engine type for progressive pipeline creation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgressiveEngineType {
    SST,
    HELIX,
    VIPER,
    SWIFT,
    NOVA,
    RAPTOR,
}

impl ProgressiveEngineType {
    /// Convert from string (case insensitive)
    pub fn from_str_uppercase(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "SST" => Some(Self::SST),
            "HELIX" => Some(Self::HELIX),
            "VIPER" => Some(Self::VIPER),
            "SWIFT" => Some(Self::SWIFT),
            "NOVA" => Some(Self::NOVA),
            "RAPTOR" => Some(Self::RAPTOR),
            _ => None,
        }
    }
}

/// Quantization stage configuration for pipeline creation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineStage {
    Binary,
    INT8,
    PQ4,
    PQ8,
    FP16,
    FP32,
}

impl PipelineStage {
    /// Convert from RL planner's QuantizationStage
    pub fn from_rl_stage(stage: &crate::query::rl_planner::action::QuantizationStage) -> Self {
        use crate::query::rl_planner::action::QuantizationStage;
        match stage {
            QuantizationStage::Binary => Self::Binary,
            QuantizationStage::INT8 => Self::INT8,
            QuantizationStage::PQ4 => Self::PQ4,
            QuantizationStage::PQ8 => Self::PQ8,
            QuantizationStage::FP16 => Self::FP16,
            QuantizationStage::FP32 => Self::FP32,
        }
    }
}

/// Factory for creating progressive search pipelines
pub struct ProgressivePipelineFactory {
    quantization_engine: Arc<UnifiedQuantizationEngine>,
    distance_compute: Arc<UnifiedDistanceCompute>,
}

impl ProgressivePipelineFactory {
    /// Create a new factory with required engines
    pub fn new(
        quantization_engine: Arc<UnifiedQuantizationEngine>,
        distance_compute: Arc<UnifiedDistanceCompute>,
    ) -> Self {
        Self {
            quantization_engine,
            distance_compute,
        }
    }

    /// Create a progressive pipeline for the specified engine with given stages
    ///
    /// # Arguments
    /// * `engine_type` - The storage engine type
    /// * `stages` - The quantization stages to include (in order)
    /// * `hamming_threshold` - Threshold for binary stage filtering (0.0-1.0)
    ///
    /// # Returns
    /// A configured ProgressiveSearchCoordinator with engine-specific stages
    pub fn create_pipeline(
        &self,
        engine_type: ProgressiveEngineType,
        stages: &[PipelineStage],
        hamming_threshold: f32,
    ) -> ProgressiveSearchCoordinator {
        match engine_type {
            ProgressiveEngineType::SST => self.create_sst_pipeline(stages, hamming_threshold),
            ProgressiveEngineType::HELIX => self.create_helix_pipeline(stages, hamming_threshold),
            ProgressiveEngineType::VIPER => self.create_viper_pipeline(stages, hamming_threshold),
            #[cfg(feature = "experimental-engines")]
            ProgressiveEngineType::SWIFT => self.create_swift_pipeline(stages, hamming_threshold),
            // SWIFT requires `experimental-engines`; empty pipeline without it.
            #[cfg(not(feature = "experimental-engines"))]
            ProgressiveEngineType::SWIFT => ProgressiveSearchCoordinator::new(),
            ProgressiveEngineType::NOVA => self.create_nova_pipeline(stages, hamming_threshold),
            #[cfg(feature = "experimental-engines")]
            ProgressiveEngineType::RAPTOR => self.create_raptor_pipeline(stages, hamming_threshold),
            // RAPTOR requires `experimental-engines`; unreachable without it, so
            // return an empty pipeline to keep the match exhaustive.
            #[cfg(not(feature = "experimental-engines"))]
            ProgressiveEngineType::RAPTOR => ProgressiveSearchCoordinator::new(),
        }
    }

    /// Create a default pipeline for an engine (Binary -> INT8 -> FP32)
    pub fn create_default_pipeline(
        &self,
        engine_type: ProgressiveEngineType,
    ) -> ProgressiveSearchCoordinator {
        let default_stages = vec![
            PipelineStage::Binary,
            PipelineStage::INT8,
            PipelineStage::FP32,
        ];
        self.create_pipeline(engine_type, &default_stages, 0.7)
    }

    /// Create pipeline from RL action's quantization stages
    pub fn create_from_action(
        &self,
        engine_type: ProgressiveEngineType,
        action: &crate::query::rl_planner::action::ExecutionAction,
    ) -> ProgressiveSearchCoordinator {
        let stages: Vec<PipelineStage> = action
            .quantization_stages
            .iter()
            .map(PipelineStage::from_rl_stage)
            .collect();

        // Use 0.7 as default hamming threshold (70% similarity required)
        self.create_pipeline(engine_type, &stages, 0.7)
    }

    // Engine-specific pipeline creators

    fn create_sst_pipeline(
        &self,
        stages: &[PipelineStage],
        hamming_threshold: f32,
    ) -> ProgressiveSearchCoordinator {
        use crate::storage::engines::sst::progressive_stages::*;

        let mut coordinator = ProgressiveSearchCoordinator::new();

        for stage in stages {
            let boxed_stage: Box<dyn ProgressiveSearchStage> = match stage {
                PipelineStage::Binary => Box::new(SstBinaryStage::new(
                    hamming_threshold,
                    self.quantization_engine.clone(),
                )),
                PipelineStage::INT8 => Box::new(SstInt8Stage::new(self.distance_compute.clone())),
                PipelineStage::FP32 => Box::new(SstFp32Stage::new(self.distance_compute.clone())),
                // FP16 and PQ stages not yet implemented for SST, use FP32 as fallback
                PipelineStage::FP16 | PipelineStage::PQ4 | PipelineStage::PQ8 => {
                    Box::new(SstFp32Stage::new(self.distance_compute.clone()))
                }
            };
            coordinator = coordinator.add_stage(boxed_stage);
        }

        coordinator
    }

    fn create_helix_pipeline(
        &self,
        stages: &[PipelineStage],
        hamming_threshold: f32,
    ) -> ProgressiveSearchCoordinator {
        use crate::storage::engines::helix::progressive_stages::*;

        let mut coordinator = ProgressiveSearchCoordinator::new();

        for stage in stages {
            let boxed_stage: Box<dyn ProgressiveSearchStage> = match stage {
                PipelineStage::Binary => Box::new(HelixBinaryStage::new(
                    hamming_threshold,
                    self.quantization_engine.clone(),
                )),
                PipelineStage::INT8 => Box::new(HelixInt8Stage::new(self.distance_compute.clone())),
                PipelineStage::FP32 => Box::new(HelixFp32Stage::new(self.distance_compute.clone())),
                PipelineStage::FP16 | PipelineStage::PQ4 | PipelineStage::PQ8 => {
                    Box::new(HelixFp32Stage::new(self.distance_compute.clone()))
                }
            };
            coordinator = coordinator.add_stage(boxed_stage);
        }

        coordinator
    }

    fn create_viper_pipeline(
        &self,
        stages: &[PipelineStage],
        hamming_threshold: f32,
    ) -> ProgressiveSearchCoordinator {
        use crate::storage::engines::viper::progressive_stages::*;

        let mut coordinator = ProgressiveSearchCoordinator::new();

        for stage in stages {
            let boxed_stage: Box<dyn ProgressiveSearchStage> = match stage {
                PipelineStage::Binary => Box::new(ViperBinaryStage::new(
                    hamming_threshold,
                    self.quantization_engine.clone(),
                )),
                PipelineStage::INT8 => Box::new(ViperInt8Stage::new(self.distance_compute.clone())),
                PipelineStage::FP32 => Box::new(ViperFp32Stage::new(self.distance_compute.clone())),
                PipelineStage::FP16 | PipelineStage::PQ4 | PipelineStage::PQ8 => {
                    Box::new(ViperFp32Stage::new(self.distance_compute.clone()))
                }
            };
            coordinator = coordinator.add_stage(boxed_stage);
        }

        coordinator
    }

    #[cfg(feature = "experimental-engines")]
    fn create_swift_pipeline(
        &self,
        stages: &[PipelineStage],
        hamming_threshold: f32,
    ) -> ProgressiveSearchCoordinator {
        use crate::storage::engines::swift::progressive_stages::*;

        let mut coordinator = ProgressiveSearchCoordinator::new();

        for stage in stages {
            let boxed_stage: Box<dyn ProgressiveSearchStage> = match stage {
                PipelineStage::Binary => Box::new(SwiftBinaryStage::new(
                    hamming_threshold,
                    self.quantization_engine.clone(),
                )),
                PipelineStage::INT8 => Box::new(SwiftInt8Stage::new(self.distance_compute.clone())),
                PipelineStage::FP32 => Box::new(SwiftFp32Stage::new(self.distance_compute.clone())),
                PipelineStage::FP16 | PipelineStage::PQ4 | PipelineStage::PQ8 => {
                    Box::new(SwiftFp32Stage::new(self.distance_compute.clone()))
                }
            };
            coordinator = coordinator.add_stage(boxed_stage);
        }

        coordinator
    }

    fn create_nova_pipeline(
        &self,
        stages: &[PipelineStage],
        hamming_threshold: f32,
    ) -> ProgressiveSearchCoordinator {
        use crate::storage::engines::nova::progressive_stages::*;

        let mut coordinator = ProgressiveSearchCoordinator::new();

        for stage in stages {
            let boxed_stage: Box<dyn ProgressiveSearchStage> = match stage {
                PipelineStage::Binary => Box::new(NovaBinaryStage::new(
                    hamming_threshold,
                    self.quantization_engine.clone(),
                )),
                PipelineStage::INT8 => Box::new(NovaInt8Stage::new(self.distance_compute.clone())),
                PipelineStage::FP32 => Box::new(NovaFp32Stage::new(self.distance_compute.clone())),
                PipelineStage::FP16 | PipelineStage::PQ4 | PipelineStage::PQ8 => {
                    Box::new(NovaFp32Stage::new(self.distance_compute.clone()))
                }
            };
            coordinator = coordinator.add_stage(boxed_stage);
        }

        coordinator
    }

    #[cfg(feature = "experimental-engines")]
    fn create_raptor_pipeline(
        &self,
        stages: &[PipelineStage],
        hamming_threshold: f32,
    ) -> ProgressiveSearchCoordinator {
        use crate::storage::engines::raptor::progressive_stages::*;

        let mut coordinator = ProgressiveSearchCoordinator::new();

        for stage in stages {
            let boxed_stage: Box<dyn ProgressiveSearchStage> = match stage {
                PipelineStage::Binary => Box::new(RaptorBinaryStage::new(
                    hamming_threshold,
                    self.quantization_engine.clone(),
                )),
                PipelineStage::INT8 => {
                    Box::new(RaptorInt8Stage::new(self.distance_compute.clone()))
                }
                PipelineStage::FP32 => {
                    Box::new(RaptorFp32Stage::new(self.distance_compute.clone()))
                }
                PipelineStage::FP16 | PipelineStage::PQ4 | PipelineStage::PQ8 => {
                    Box::new(RaptorFp32Stage::new(self.distance_compute.clone()))
                }
            };
            coordinator = coordinator.add_stage(boxed_stage);
        }

        coordinator
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::quantization::quantization_engine::{CodebookStore, InMemoryCodebookStore};
    use proximadb_distance_kernel::DistanceMetric;

    fn create_test_factory() -> ProgressivePipelineFactory {
        let dist_compute = Arc::new(UnifiedDistanceCompute::new(DistanceMetric::Cosine));
        let codebook_store: Arc<dyn CodebookStore> = Arc::new(InMemoryCodebookStore::new());
        let quant_engine = Arc::new(UnifiedQuantizationEngine::new(
            dist_compute.clone(),
            codebook_store,
        ));
        ProgressivePipelineFactory::new(quant_engine, dist_compute)
    }

    #[test]
    fn test_factory_create_default_pipeline() {
        let factory = create_test_factory();

        let mut engines = vec![
            ProgressiveEngineType::SST,
            ProgressiveEngineType::HELIX,
            ProgressiveEngineType::VIPER,
            ProgressiveEngineType::SWIFT,
            ProgressiveEngineType::NOVA,
        ];
        // RAPTOR's pipeline is `experimental-engines`-gated (it needs AXIS clustering);
        // only assert its 3-stage default when that feature is on.
        #[cfg(feature = "experimental-engines")]
        engines.push(ProgressiveEngineType::RAPTOR);

        for engine in &engines {
            let pipeline = factory.create_default_pipeline(*engine);
            assert_eq!(
                pipeline.stage_count(),
                3,
                "Default pipeline for {:?} should have 3 stages",
                engine
            );
        }
    }

    #[test]
    fn test_factory_create_custom_pipeline() {
        let factory = create_test_factory();
        let stages = vec![
            PipelineStage::Binary,
            PipelineStage::INT8,
            PipelineStage::PQ8,
            PipelineStage::FP32,
        ];

        let pipeline = factory.create_pipeline(ProgressiveEngineType::HELIX, &stages, 0.8);
        assert_eq!(pipeline.stage_count(), 4);
    }

    #[test]
    fn test_engine_type_from_str_uppercase() {
        assert_eq!(
            ProgressiveEngineType::from_str_uppercase("sst"),
            Some(ProgressiveEngineType::SST)
        );
        assert_eq!(
            ProgressiveEngineType::from_str_uppercase("HELIX"),
            Some(ProgressiveEngineType::HELIX)
        );
        assert_eq!(ProgressiveEngineType::from_str_uppercase("unknown"), None);
    }
}
