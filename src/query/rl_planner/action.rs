//! Execution Action Definitions
//!
//! Defines the action space for the RL planner, including index strategies,
//! search modes, quantization pipelines, and engine-specific optimizations.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Unique identifier for discretized actions
pub type ActionId = u32;

/// Primary index strategy selection
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub enum IndexStrategy {
    /// Hierarchical Navigable Small World graph
    HNSW {
        /// Number of connections per layer (typically 8-64)
        m: u8,
        /// Search expansion factor (typically 50-200)
        ef_search: u16,
    },
    /// Inverted File index with clustering
    IVF {
        /// Number of clusters to probe (typically 1-32)
        n_probe: u16,
    },
    /// Locality Sensitive Hashing
    LSH {
        /// Number of hash tables
        n_tables: u8,
        /// Number of hash functions per table
        n_hashes: u8,
    },
    /// Annoy (Approximate Nearest Neighbors Oh Yeah)
    Annoy {
        /// Number of trees
        n_trees: u8,
        /// Search parameter (-1 for auto)
        search_k: i32,
    },
    /// Product Quantization
    PQ {
        /// Number of subvectors
        n_subvectors: u8,
        /// Bits per subvector code
        bits: u8,
    },
    /// No index, direct scan
    #[default]
    DirectScan,
}

impl fmt::Display for IndexStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HNSW { m, ef_search } => write!(f, "HNSW(m={}, ef={})", m, ef_search),
            Self::IVF { n_probe } => write!(f, "IVF(nprobe={})", n_probe),
            Self::LSH { n_tables, n_hashes } => {
                write!(f, "LSH(tables={}, hashes={})", n_tables, n_hashes)
            }
            Self::Annoy { n_trees, search_k } => {
                write!(f, "Annoy(trees={}, k={})", n_trees, search_k)
            }
            Self::PQ { n_subvectors, bits } => write!(f, "PQ(sub={}, bits={})", n_subvectors, bits),
            Self::DirectScan => write!(f, "DirectScan"),
        }
    }
}

/// Search mode selection
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub enum SearchModeAction {
    /// Exact search (100% recall, slowest)
    #[default]
    Exact,
    /// Approximate search with expansion factor
    Approximate {
        /// Expansion factor for candidate set (e.g., 10 = search 10x top_k)
        expansion_factor: f32,
    },
    /// Adaptive: starts approximate, switches to exact if needed
    Adaptive {
        /// Threshold for switching to exact (collection size)
        threshold: u32,
    },
}

impl fmt::Display for SearchModeAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Exact => write!(f, "Exact"),
            Self::Approximate { expansion_factor } => {
                write!(f, "Approximate(exp={})", expansion_factor)
            }
            Self::Adaptive { threshold } => write!(f, "Adaptive(thresh={})", threshold),
        }
    }
}

/// Quantization stage in progressive pipeline
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QuantizationStage {
    /// 1-bit binary quantization (32x compression, ~70% recall)
    Binary,
    /// 8-bit integer quantization (4x compression, ~95% recall)
    INT8,
    /// 4-bit product quantization (8x compression, ~85% recall)
    PQ4,
    /// 8-bit product quantization (4-32x compression, ~90% recall)
    PQ8,
    /// 16-bit floating point
    FP16,
    /// Full 32-bit floating point (no compression, 100% recall)
    FP32,
}

impl fmt::Display for QuantizationStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Binary => write!(f, "Binary"),
            Self::INT8 => write!(f, "INT8"),
            Self::PQ4 => write!(f, "PQ4"),
            Self::PQ8 => write!(f, "PQ8"),
            Self::FP16 => write!(f, "FP16"),
            Self::FP32 => write!(f, "FP32"),
        }
    }
}

impl QuantizationStage {
    /// Quality rank where higher = higher precision/recall. Ordering is
    /// derived from the per-variant doc comments above:
    /// `Binary ~70% < PQ4 ~85% < PQ8 ~90% < INT8 ~95% < FP16 < FP32 100%`.
    /// Used by the object-economy floor enforcement to compare current
    /// search precision against a required minimum.
    pub fn quality_rank(self) -> u8 {
        match self {
            Self::Binary => 0,
            Self::PQ4 => 1,
            Self::PQ8 => 2,
            Self::INT8 => 3,
            Self::FP16 => 4,
            Self::FP32 => 5,
        }
    }

    /// Whether this stage demands exact precision (FP16/FP32) — used by the
    /// object-economy floor enforcement.
    pub fn requires_exact_final_stage(self) -> bool {
        self.quality_rank() >= Self::FP16.quality_rank()
    }
}

/// Block pruning configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub enum BlockPruneConfig {
    /// No block pruning
    #[default]
    Off,
    /// Prune sqrt(N) blocks
    Sqrt,
    /// Prune based on ratio (e.g., 0.5 = keep top 50% of blocks)
    Ratio(f32),
    /// Prune based on centroid distance threshold
    CentroidDistance {
        /// Maximum distance ratio from nearest centroid
        threshold: f32,
    },
    /// Zone map based pruning
    ZoneMap,
}

impl fmt::Display for BlockPruneConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Off => write!(f, "Off"),
            Self::Sqrt => write!(f, "Sqrt"),
            Self::Ratio(r) => write!(f, "Ratio({})", r),
            Self::CentroidDistance { threshold } => write!(f, "Centroid(thresh={})", threshold),
            Self::ZoneMap => write!(f, "ZoneMap"),
        }
    }
}

/// Parallelism configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParallelismConfig {
    /// Number of threads to use
    pub num_threads: u8,
    /// Enable SIMD acceleration
    pub enable_simd: bool,
    /// Batch size for parallel processing
    pub batch_size: u32,
}

impl Default for ParallelismConfig {
    fn default() -> Self {
        Self {
            num_threads: num_cpus::get() as u8,
            enable_simd: true,
            batch_size: 1024,
        }
    }
}

/// Object economy routing actions
///
/// Encodes decisions for using object economy metadata
/// for cost-aware routing and tier-specific optimization.
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObjectEconomyAction {
    /// Whether object economy routing is enabled
    pub enabled: bool,
    /// Use centroid-based routing for block selection
    pub use_centroid_routing: bool,
    /// Use Z-order code based pruning
    pub use_zorder_pruning: bool,
    /// Maximum blocks to scan (None = no limit)
    pub max_blocks_to_scan: Option<u32>,
    /// Request-side minimum quality floor: when set, this query must run at
    /// least at this quantization stage (e.g. FP16 for quality-critical
    /// reranking). Distinct from the tier-side
    /// [`ObjectEconomyQuantizationCeiling`](crate::catalog::tenant_tier::ObjectEconomyQuantizationCeiling),
    /// which caps the *maximum* precision a tier is allowed to request.
    pub quantization_floor: Option<QuantizationStage>,
}

impl fmt::Display for ObjectEconomyAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if !self.enabled {
            return write!(f, "OE:OFF");
        }

        let mut parts = vec!["OE:ON".to_string()];

        if self.use_centroid_routing {
            parts.push("Centroid".to_string());
        }

        if self.use_zorder_pruning {
            parts.push("ZOrder".to_string());
        }

        if let Some(max_blocks) = self.max_blocks_to_scan {
            parts.push(format!("MaxBlocks({})", max_blocks));
        }

        if let Some(floor) = &self.quantization_floor {
            parts.push(format!("Floor({})", floor));
        }

        write!(f, "{}", parts.join("+"))
    }
}

impl ObjectEconomyAction {
    /// Create enabled object economy action with centroid routing
    pub fn with_centroid_routing(max_blocks: Option<u32>) -> Self {
        Self {
            enabled: true,
            use_centroid_routing: true,
            use_zorder_pruning: false,
            max_blocks_to_scan: max_blocks,
            quantization_floor: None,
        }
    }

    /// Create enabled object economy action with Z-order pruning
    pub fn with_zorder_pruning(max_blocks: Option<u32>) -> Self {
        Self {
            enabled: true,
            use_centroid_routing: false,
            use_zorder_pruning: true,
            max_blocks_to_scan: max_blocks,
            quantization_floor: None,
        }
    }

    /// Create action with quantization floor (for tier routing)
    pub fn with_quantization_floor(floor: QuantizationStage) -> Self {
        Self {
            enabled: true,
            use_centroid_routing: false,
            use_zorder_pruning: false,
            max_blocks_to_scan: None,
            quantization_floor: Some(floor),
        }
    }
}

/// Complete execution action combining all optimization choices
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExecutionAction {
    /// Primary index strategy (can be None for direct scan)
    pub index_strategy: Option<IndexStrategy>,
    /// Secondary index for two-stage search (e.g., IVF -> HNSW)
    pub secondary_index: Option<IndexStrategy>,
    /// Search mode
    pub search_mode: SearchModeAction,
    /// Quantization pipeline (stages executed in order)
    pub quantization_stages: Vec<QuantizationStage>,
    /// Block pruning configuration
    pub block_pruning: BlockPruneConfig,
    /// Enable zone map filtering
    pub zone_map_enabled: bool,
    /// Enable bloom filter pre-filtering
    pub bloom_filter_enabled: bool,
    /// Parallelism settings
    pub parallelism: ParallelismConfig,
    /// Object economy routing actions
    pub object_economy: ObjectEconomyAction,
}

impl Default for ExecutionAction {
    fn default() -> Self {
        Self {
            index_strategy: Some(IndexStrategy::DirectScan),
            secondary_index: None,
            search_mode: SearchModeAction::Exact,
            quantization_stages: vec![QuantizationStage::FP32],
            block_pruning: BlockPruneConfig::Off,
            zone_map_enabled: false,
            bloom_filter_enabled: false,
            parallelism: ParallelismConfig::default(),
            object_economy: ObjectEconomyAction::default(),
        }
    }
}

impl ExecutionAction {
    /// Create action with HNSW index
    pub fn with_hnsw(ef_search: u16) -> Self {
        Self {
            index_strategy: Some(IndexStrategy::HNSW { m: 16, ef_search }),
            search_mode: SearchModeAction::Approximate {
                expansion_factor: 1.0,
            },
            ..Default::default()
        }
    }

    /// Create action with IVF index
    pub fn with_ivf(n_probe: u16) -> Self {
        Self {
            index_strategy: Some(IndexStrategy::IVF { n_probe }),
            search_mode: SearchModeAction::Approximate {
                expansion_factor: 1.0,
            },
            ..Default::default()
        }
    }

    /// Create action with progressive quantization pipeline
    pub fn with_progressive_quantization() -> Self {
        Self {
            quantization_stages: vec![
                QuantizationStage::Binary,
                QuantizationStage::INT8,
                QuantizationStage::FP32,
            ],
            search_mode: SearchModeAction::Approximate {
                expansion_factor: 2.0,
            },
            ..Default::default()
        }
    }

    /// Create action for two-stage search (IVF coarse + HNSW fine)
    pub fn with_two_stage_index(coarse: IndexStrategy, fine: IndexStrategy) -> Self {
        Self {
            index_strategy: Some(coarse),
            secondary_index: Some(fine),
            search_mode: SearchModeAction::Approximate {
                expansion_factor: 1.5,
            },
            ..Default::default()
        }
    }

    /// Add block pruning to action
    pub fn with_block_pruning(mut self, config: BlockPruneConfig) -> Self {
        self.block_pruning = config;
        self
    }

    /// Enable zone map filtering
    pub fn with_zone_map(mut self) -> Self {
        self.zone_map_enabled = true;
        self
    }

    /// Enable bloom filter
    pub fn with_bloom_filter(mut self) -> Self {
        self.bloom_filter_enabled = true;
        self
    }

    /// Set object economy routing action
    pub fn with_object_economy(mut self, oe: ObjectEconomyAction) -> Self {
        self.object_economy = oe;
        self
    }

    /// Compute unique action ID for discrete action space
    pub fn to_action_id(&self) -> ActionId {
        let mut id: ActionId = 0;

        // Index strategy (0-9)
        id += match &self.index_strategy {
            None => 0,
            Some(IndexStrategy::DirectScan) => 1,
            Some(IndexStrategy::HNSW { ef_search, .. }) => {
                2 + (*ef_search / 50).min(2) as ActionId // 3 levels: 50, 100, 150+
            }
            Some(IndexStrategy::IVF { n_probe }) => {
                5 + (*n_probe / 8).min(2) as ActionId // 3 levels: 8, 16, 24+
            }
            Some(IndexStrategy::LSH { .. }) => 8,
            Some(IndexStrategy::Annoy { .. }) => 9,
            Some(IndexStrategy::PQ { .. }) => 10,
        };

        // Search mode (0-4) shifted by 4 bits
        id += match &self.search_mode {
            SearchModeAction::Exact => 0,
            SearchModeAction::Approximate { expansion_factor } => {
                1 + (*expansion_factor as u32).min(3)
            }
            SearchModeAction::Adaptive { .. } => 5,
        } << 4;

        // Quantization (0-7) shifted by 8 bits
        let quant_code = if self.quantization_stages.is_empty() {
            0
        } else if self.quantization_stages.len() == 1 {
            match self.quantization_stages[0] {
                QuantizationStage::FP32 => 1,
                QuantizationStage::FP16 => 2,
                QuantizationStage::INT8 => 3,
                QuantizationStage::Binary => 4,
                _ => 5,
            }
        } else {
            // Progressive pipeline
            6 + self.quantization_stages.len().min(2) as ActionId
        };
        id += quant_code << 8;

        // Block pruning (0-4) shifted by 12 bits
        id += match &self.block_pruning {
            BlockPruneConfig::Off => 0,
            BlockPruneConfig::Sqrt => 1,
            BlockPruneConfig::Ratio(_) => 2,
            BlockPruneConfig::CentroidDistance { .. } => 3,
            BlockPruneConfig::ZoneMap => 4,
        } << 12;

        // Bloom filter (1 bit) shifted by 16 bits
        id += if self.bloom_filter_enabled { 1 } else { 0 } << 16;

        id
    }

    /// Create action from discrete action ID
    pub fn from_action_id(id: ActionId) -> Self {
        let index_code = id & 0xF;
        let search_code = (id >> 4) & 0xF;
        let quant_code = (id >> 8) & 0xF;
        let prune_code = (id >> 12) & 0xF;
        let bloom = (id >> 16) & 0x1;

        let index_strategy = match index_code {
            0 => None,
            1 => Some(IndexStrategy::DirectScan),
            2 => Some(IndexStrategy::HNSW {
                m: 16,
                ef_search: 50,
            }),
            3 => Some(IndexStrategy::HNSW {
                m: 16,
                ef_search: 100,
            }),
            4 => Some(IndexStrategy::HNSW {
                m: 16,
                ef_search: 200,
            }),
            5 => Some(IndexStrategy::IVF { n_probe: 8 }),
            6 => Some(IndexStrategy::IVF { n_probe: 16 }),
            7 => Some(IndexStrategy::IVF { n_probe: 32 }),
            8 => Some(IndexStrategy::LSH {
                n_tables: 10,
                n_hashes: 8,
            }),
            9 => Some(IndexStrategy::Annoy {
                n_trees: 10,
                search_k: -1,
            }),
            10 => Some(IndexStrategy::PQ {
                n_subvectors: 8,
                bits: 8,
            }),
            _ => Some(IndexStrategy::DirectScan),
        };

        let search_mode = match search_code {
            0 => SearchModeAction::Exact,
            1 => SearchModeAction::Approximate {
                expansion_factor: 1.0,
            },
            2 => SearchModeAction::Approximate {
                expansion_factor: 2.0,
            },
            3 => SearchModeAction::Approximate {
                expansion_factor: 3.0,
            },
            4 => SearchModeAction::Approximate {
                expansion_factor: 4.0,
            },
            _ => SearchModeAction::Adaptive { threshold: 10000 },
        };

        let quantization_stages = match quant_code {
            0 => vec![],
            1 => vec![QuantizationStage::FP32],
            2 => vec![QuantizationStage::FP16],
            3 => vec![QuantizationStage::INT8],
            4 => vec![QuantizationStage::Binary],
            5 => vec![QuantizationStage::PQ8],
            6 => vec![QuantizationStage::Binary, QuantizationStage::FP32],
            7 => vec![
                QuantizationStage::Binary,
                QuantizationStage::INT8,
                QuantizationStage::FP32,
            ],
            _ => vec![
                QuantizationStage::Binary,
                QuantizationStage::INT8,
                QuantizationStage::PQ8,
                QuantizationStage::FP32,
            ],
        };

        let block_pruning = match prune_code {
            0 => BlockPruneConfig::Off,
            1 => BlockPruneConfig::Sqrt,
            2 => BlockPruneConfig::Ratio(0.5),
            3 => BlockPruneConfig::CentroidDistance { threshold: 1.5 },
            _ => BlockPruneConfig::ZoneMap,
        };
        let zone_map_enabled = matches!(block_pruning, BlockPruneConfig::ZoneMap);

        Self {
            index_strategy,
            secondary_index: None,
            search_mode,
            quantization_stages,
            block_pruning,
            zone_map_enabled,
            bloom_filter_enabled: bloom == 1,
            parallelism: ParallelismConfig::default(),
            object_economy: ObjectEconomyAction::default(), // Disabled by default for backward compat
        }
    }

    /// Describe action for logging
    pub fn describe(&self) -> String {
        let mut parts = Vec::new();

        if let Some(idx) = &self.index_strategy {
            parts.push(format!("Index: {}", idx));
        }
        if let Some(sec) = &self.secondary_index {
            parts.push(format!("Secondary: {}", sec));
        }
        parts.push(format!("Mode: {}", self.search_mode));

        if !self.quantization_stages.is_empty() {
            let stages: Vec<_> = self
                .quantization_stages
                .iter()
                .map(|s| s.to_string())
                .collect();
            parts.push(format!("Quant: {}", stages.join("→")));
        }

        if !matches!(self.block_pruning, BlockPruneConfig::Off) {
            parts.push(format!("Prune: {}", self.block_pruning));
        }

        if self.bloom_filter_enabled {
            parts.push("Bloom: ON".to_string());
        }

        // Include object economy action if enabled
        if self.object_economy.enabled {
            parts.push(format!("{}", self.object_economy));
        }

        parts.join(", ")
    }
}

/// Action space for systematic exploration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionSpace {
    /// All possible discrete actions
    pub actions: Vec<ExecutionAction>,
}

impl Default for ActionSpace {
    fn default() -> Self {
        Self {
            actions: Self::default_actions(),
        }
    }
}

impl ActionSpace {
    /// Get default actions (used for Default implementation)
    fn default_actions() -> Vec<ExecutionAction> {
        vec![
            // Baseline
            ExecutionAction::default(),
            // HNSW variants
            ExecutionAction::with_hnsw(50),
            ExecutionAction::with_hnsw(100),
            ExecutionAction::with_hnsw(200),
            // IVF variants
            ExecutionAction::with_ivf(8),
            ExecutionAction::with_ivf(16),
            ExecutionAction::with_ivf(32),
            // Progressive quantization
            ExecutionAction::with_progressive_quantization(),
        ]
    }

    /// Create action space for a given storage engine
    pub fn for_engine(engine: &str) -> Self {
        let actions = match engine.to_uppercase().as_str() {
            "SST" => Self::sst_actions(),
            "HELIX" => Self::helix_actions(),
            "VIPER" => Self::viper_actions(),
            "SWIFT" => Self::swift_actions(),
            "NOVA" => Self::nova_actions(),
            "RAPTOR" => Self::raptor_actions(),
            _ => Self::default_actions(),
        };

        Self { actions }
    }

    /// SST engine actions
    fn sst_actions() -> Vec<ExecutionAction> {
        vec![
            // 1. DirectScan + FP32 (baseline)
            ExecutionAction::default(),
            // 2. HNSW(ef=50) + FP32
            ExecutionAction::with_hnsw(50),
            // 3. HNSW(ef=100) + FP32
            ExecutionAction::with_hnsw(100),
            // 4. HNSW(ef=200) + FP32
            ExecutionAction::with_hnsw(200),
            // 5. IVF(nprobe=4) + FP32
            ExecutionAction::with_ivf(4),
            // 6. IVF(nprobe=16) + FP32
            ExecutionAction::with_ivf(16),
            // 7. IVF(nprobe=32) + FP32
            ExecutionAction::with_ivf(32),
            // 8. Bloom + BlockPrune + FP32
            ExecutionAction::default()
                .with_bloom_filter()
                .with_block_pruning(BlockPruneConfig::Sqrt),
            // 9. HNSW + Bloom + BlockPrune
            ExecutionAction::with_hnsw(100)
                .with_bloom_filter()
                .with_block_pruning(BlockPruneConfig::Sqrt),
            // 10. Progressive(Binary→INT8→FP32)
            ExecutionAction::with_progressive_quantization(),
        ]
    }

    /// HELIX engine actions (Hilbert curve + PCA)
    fn helix_actions() -> Vec<ExecutionAction> {
        vec![
            // 1. DirectScan + FP32 (baseline)
            ExecutionAction::default(),
            // 2. HNSW + ZoneMap + FP32
            ExecutionAction::with_hnsw(100).with_zone_map(),
            // 3. Progressive(Binary→INT8→PQ→FP32)
            ExecutionAction {
                quantization_stages: vec![
                    QuantizationStage::Binary,
                    QuantizationStage::INT8,
                    QuantizationStage::PQ8,
                    QuantizationStage::FP32,
                ],
                search_mode: SearchModeAction::Approximate {
                    expansion_factor: 2.0,
                },
                ..Default::default()
            },
            // 4. HilbertPrune + Progressive
            ExecutionAction {
                block_pruning: BlockPruneConfig::CentroidDistance { threshold: 1.5 },
                quantization_stages: vec![
                    QuantizationStage::Binary,
                    QuantizationStage::INT8,
                    QuantizationStage::FP32,
                ],
                ..Default::default()
            },
            // 5. PCA + HilbertPrune + FP32
            ExecutionAction {
                block_pruning: BlockPruneConfig::CentroidDistance { threshold: 2.0 },
                ..Default::default()
            },
            // 6. IVF + ZoneMap + INT8
            ExecutionAction::with_ivf(16)
                .with_zone_map()
                .with_block_pruning(BlockPruneConfig::ZoneMap),
            // 7. LSH + Progressive
            ExecutionAction {
                index_strategy: Some(IndexStrategy::LSH {
                    n_tables: 10,
                    n_hashes: 8,
                }),
                quantization_stages: vec![
                    QuantizationStage::Binary,
                    QuantizationStage::INT8,
                    QuantizationStage::FP32,
                ],
                ..Default::default()
            },
            // 8. Full Progressive (5-stage)
            ExecutionAction {
                quantization_stages: vec![
                    QuantizationStage::Binary,
                    QuantizationStage::INT8,
                    QuantizationStage::PQ4,
                    QuantizationStage::PQ8,
                    QuantizationStage::FP32,
                ],
                search_mode: SearchModeAction::Approximate {
                    expansion_factor: 3.0,
                },
                block_pruning: BlockPruneConfig::CentroidDistance { threshold: 2.0 },
                ..Default::default()
            },
        ]
    }

    /// VIPER engine actions (Columnar Parquet)
    fn viper_actions() -> Vec<ExecutionAction> {
        vec![
            // 1. RowGroupScan + FP32 (baseline)
            ExecutionAction::default(),
            // 2. RowGroupPrune + FP32
            ExecutionAction::default().with_block_pruning(BlockPruneConfig::Ratio(0.5)),
            // 3. ColumnProjection + FP32
            ExecutionAction::default(),
            // 4. Binary Pre-filter + FP32
            ExecutionAction {
                quantization_stages: vec![QuantizationStage::Binary, QuantizationStage::FP32],
                ..Default::default()
            },
            // 5. INT8 Columnar + FP32
            ExecutionAction {
                quantization_stages: vec![QuantizationStage::INT8, QuantizationStage::FP32],
                ..Default::default()
            },
            // 6. HNSW + RowGroupPrune
            ExecutionAction::with_hnsw(100).with_block_pruning(BlockPruneConfig::Ratio(0.3)),
            // 7. IVF + Columnar + INT8
            ExecutionAction {
                index_strategy: Some(IndexStrategy::IVF { n_probe: 16 }),
                quantization_stages: vec![QuantizationStage::INT8, QuantizationStage::FP32],
                ..Default::default()
            },
        ]
    }

    /// SWIFT engine actions (Ultra-low latency)
    fn swift_actions() -> Vec<ExecutionAction> {
        vec![
            // 1. InMemory + FP32 (baseline)
            ExecutionAction::default(),
            // 2. InMemory + INT8
            ExecutionAction {
                quantization_stages: vec![QuantizationStage::INT8],
                ..Default::default()
            },
            // 3. Progressive(Binary→INT8→FP32)
            ExecutionAction::with_progressive_quantization(),
            // 4. HNSW + InMemory
            ExecutionAction::with_hnsw(50),
            // 5. Parallel + Progressive
            ExecutionAction {
                quantization_stages: vec![
                    QuantizationStage::Binary,
                    QuantizationStage::INT8,
                    QuantizationStage::FP32,
                ],
                parallelism: ParallelismConfig {
                    num_threads: 8,
                    enable_simd: true,
                    batch_size: 512,
                },
                ..Default::default()
            },
        ]
    }

    /// NOVA engine actions (Progressive columnar)
    fn nova_actions() -> Vec<ExecutionAction> {
        vec![
            // 1. Columnar + FP32 (baseline)
            ExecutionAction::default(),
            // 2. ZoneMap + FP32
            ExecutionAction::default().with_zone_map(),
            // 3. Progressive + ZoneMap
            ExecutionAction::with_progressive_quantization().with_zone_map(),
            // 4. IVF + Columnar
            ExecutionAction::with_ivf(16),
            // 5. Streaming + Progressive
            ExecutionAction {
                search_mode: SearchModeAction::Adaptive { threshold: 5000 },
                quantization_stages: vec![
                    QuantizationStage::Binary,
                    QuantizationStage::INT8,
                    QuantizationStage::FP32,
                ],
                ..Default::default()
            },
        ]
    }

    /// RAPTOR engine actions (Adaptive row-group)
    fn raptor_actions() -> Vec<ExecutionAction> {
        vec![
            // 1. MatrixScan + FP32 (baseline)
            ExecutionAction::default(),
            // 2. P²Matrix + FP32
            ExecutionAction::default().with_block_pruning(BlockPruneConfig::Sqrt),
            // 3. K²Matrix + Pruning
            ExecutionAction::default()
                .with_block_pruning(BlockPruneConfig::CentroidDistance { threshold: 1.5 }),
            // 4. Adaptive + Progressive
            ExecutionAction {
                search_mode: SearchModeAction::Adaptive { threshold: 10000 },
                quantization_stages: vec![
                    QuantizationStage::Binary,
                    QuantizationStage::INT8,
                    QuantizationStage::FP32,
                ],
                ..Default::default()
            },
            // 5. MultiTier + Quantized
            ExecutionAction {
                index_strategy: Some(IndexStrategy::IVF { n_probe: 16 }),
                quantization_stages: vec![QuantizationStage::INT8, QuantizationStage::FP32],
                block_pruning: BlockPruneConfig::Ratio(0.3),
                ..Default::default()
            },
        ]
    }

    /// Get number of actions in space
    pub fn len(&self) -> usize {
        self.actions.len()
    }

    /// Check if action space is empty
    pub fn is_empty(&self) -> bool {
        self.actions.is_empty()
    }

    /// Get random action
    pub fn random_action(&self) -> &ExecutionAction {
        use rand::Rng;
        let idx = rand::thread_rng().gen_range(0..self.actions.len());
        &self.actions[idx]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_action_id_roundtrip() {
        let actions = vec![
            ExecutionAction::default(),
            ExecutionAction::with_hnsw(100),
            ExecutionAction::with_ivf(16),
            ExecutionAction::with_progressive_quantization(),
        ];

        for action in actions {
            let id = action.to_action_id();
            let recovered = ExecutionAction::from_action_id(id);

            // Main properties should match
            assert_eq!(
                action.index_strategy.is_some(),
                recovered.index_strategy.is_some()
            );
            assert_eq!(action.bloom_filter_enabled, recovered.bloom_filter_enabled);
        }
    }

    #[test]
    fn test_action_space_for_engines() {
        for engine in &["SST", "HELIX", "VIPER", "SWIFT", "NOVA", "RAPTOR"] {
            let space = ActionSpace::for_engine(engine);
            assert!(!space.is_empty(), "Empty action space for {}", engine);
            assert!(space.len() >= 5, "Too few actions for {}", engine);
        }
    }

    #[test]
    fn test_action_describe() {
        let action = ExecutionAction::with_hnsw(100)
            .with_bloom_filter()
            .with_block_pruning(BlockPruneConfig::Sqrt);

        let desc = action.describe();
        assert!(desc.contains("HNSW"));
        assert!(desc.contains("Bloom"));
        assert!(desc.contains("Sqrt"));
    }

    #[test]
    fn test_progressive_quantization_action() {
        let action = ExecutionAction::with_progressive_quantization();
        assert_eq!(action.quantization_stages.len(), 3);
        assert_eq!(action.quantization_stages[0], QuantizationStage::Binary);
        assert_eq!(action.quantization_stages[1], QuantizationStage::INT8);
        assert_eq!(action.quantization_stages[2], QuantizationStage::FP32);
    }

    #[test]
    fn test_object_economy_action_default() {
        let oe = ObjectEconomyAction::default();
        assert!(!oe.enabled);
        assert!(!oe.use_centroid_routing);
        assert!(!oe.use_zorder_pruning);
        assert!(oe.max_blocks_to_scan.is_none());
        assert!(oe.quantization_floor.is_none());
    }

    #[test]
    fn test_object_economy_action_with_centroid_routing() {
        let oe = ObjectEconomyAction::with_centroid_routing(Some(100));
        assert!(oe.enabled);
        assert!(oe.use_centroid_routing);
        assert!(!oe.use_zorder_pruning);
        assert_eq!(oe.max_blocks_to_scan, Some(100));
    }

    #[test]
    fn test_object_economy_action_with_zorder_pruning() {
        let oe = ObjectEconomyAction::with_zorder_pruning(None);
        assert!(oe.enabled);
        assert!(!oe.use_centroid_routing);
        assert!(oe.use_zorder_pruning);
        assert!(oe.max_blocks_to_scan.is_none());
    }

    #[test]
    fn test_object_economy_action_with_quantization_floor() {
        let oe = ObjectEconomyAction::with_quantization_floor(QuantizationStage::INT8);
        assert!(oe.enabled);
        assert_eq!(oe.quantization_floor, Some(QuantizationStage::INT8));
    }

    #[test]
    fn test_execution_action_with_object_economy() {
        let action = ExecutionAction::default()
            .with_object_economy(ObjectEconomyAction::with_centroid_routing(Some(50)));

        assert!(action.object_economy.enabled);
        assert!(action.object_economy.use_centroid_routing);
        assert_eq!(action.object_economy.max_blocks_to_scan, Some(50));
    }

    #[test]
    fn test_action_describe_includes_object_economy() {
        let action = ExecutionAction::default()
            .with_object_economy(ObjectEconomyAction::with_centroid_routing(Some(100)));

        let desc = action.describe();
        assert!(desc.contains("OE"));
        assert!(desc.contains("Centroid"));
    }
}
