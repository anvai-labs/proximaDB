//! Storage format abstractions
//!
//! Provides Proxima block-columnar and pure columnar storage formats shared by multiple engines

pub mod arrow_block;
pub mod codebook_metadata;
pub mod columnar;
pub mod common_quantization;
pub mod proximablocks;
pub mod quantized_schema;
pub mod vector_serialization;

#[cfg(test)]
mod codebook_integration_test;

// Proxima block-columnar formats are used by SST and SWIFT (vectors are columnar-encoded within blocks)
// Pure columnar formats are used by VIPER and NOVA (Parquet-based)

pub use proximablocks::{ProximaBlockMetadata, ProximaDataBlock, SstIOLayer};

pub use columnar::{
    ColumnarOperations, FooterCache, ParquetIOLayer, ParquetQueryEngine, ParquetWriter,
};

pub use common_quantization::{
    EngineQuantizationConfig, ProximaBlockQuantizationStorage, QuantizationFileConfig,
    QuantizationTrigger, QuantizedVectorData, UnifiedQuantizedFile,
};

pub use quantized_schema::{
    PhysicalFieldSpec, QuantizedFieldDefinition, QuantizedVectorSchema,
    QuantizedVectorSchemaBuilder, SchemaStorageType, StorageMapping,
};

pub use codebook_metadata::{
    BinaryCodebook, CodebookSerializer, Int8Codebook, PqCodebook, PqTrainingConfig,
    ProximaBlockFooter, QuantizationCodebookMetadata,
};

pub use vector_serialization::VectorSerializer;

pub use arrow_block::{
    ArrowBlockConfig, ArrowBlockError, ArrowBlockIndex, ArrowBlockMetadata, ArrowBlockReader,
    ArrowBlockResult, ArrowBlockWriter, ArrowIndexEntry,
};
