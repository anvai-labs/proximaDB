/*
 * Copyright 2025 ProximaDB
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

//! Compact vector representation for AXIS indexes
//!
//! This module provides a highly efficient packed representation for vectors
//! that eliminates proto overhead and enables zero-copy access patterns.
//!
//! # Design Evolution & Key Insights
//!
//! ## Initial Observation (User Insight #1)
//! "Why store full VectorRecord with metadata, timestamps, etc. when AXIS indexes
//! only need ID and vector data? Just store raw vectors as bytemuck binary."
//!
//! ## Progressive Optimization Journey
//!
//! ### Version 1: Standard header with dimension (8 bytes overhead)
//! - Layout: [dimension:2][id_len:2][flags:1][quant_params:3][vector][id]
//! - Problem: Storing dimension and id_len for every vector
//!
//! ### Version 2: Removed id_len field (User Insight #2)
//! "Why do we need id_len when everything after fixed-size vector is the ID?"
//! - Layout: [dimension:2][flags:1][quant_info:1][vector][id]
//! - Reduced to 4 bytes overhead
//! - ID is simply: data[header + vector_size..]
//!
//! ### Version 3: External dimension storage (User Insight #3)
//! "Dimension is constant per collection - why repeat it? Store once in collection
//! cache shared with AXIS from VectorOperationsService. O(1) retrieval."
//! - Layout FP32: [vector][id] (ZERO overhead!)
//! - Layout Quantized: [quant_info:1][vector][id] (1 byte overhead)
//!
//! ### Version 4: External quantization config (User Insight #4)
//! "Quantization info is also constant per collection, may be overridden per index
//! type, and is already cached in shared collection cache. No need to store it."
//! - Layout: [vector][id] for BOTH FP32 and quantized!
//! - ZERO metadata overhead - just raw data!
//!
//! ## Memory Savings Analysis
//!
//! For 384-dimension vectors with 16-char IDs:
//! - Original VectorRecord (proto): ~1600 bytes
//! - Version 1 (8-byte header): 1560 bytes (2.5% reduction)
//! - Version 2 (4-byte header): 1556 bytes (2.75% reduction)  
//! - Version 3 (external dim): 1552 bytes FP32, 401 bytes INT8 (3% / 75% reduction)
//! - Version 4 (ultra-compact): 1552 bytes FP32, 400 bytes INT8 (3% / 75% reduction)
//!
//! ## Key Architectural Principles
//!
//! 1. **Leverage Shared State**: Collections have fixed dimension & quantization
//! 2. **Zero-Copy Access**: Use bytemuck for direct memory mapping
//! 3. **Eliminate Redundancy**: Don't store what's already known
//! 4. **Cache-Aware Design**: Collection config is in O(1) shared cache
//!
//! ## Implementation Notes
//!
//! The CompactVector below implements Version 3 (external dimension).
//! Version 4 (ultra-compact, all metadata overhead moved to collection-level
//! configuration) was prototyped but never compiled, and was removed under TD-ORPHAN-1.

use anyhow::{Result, anyhow};
use bytemuck;

/// Ultra-compact vector storage - dimension stored externally
///
/// Layout for FP32: [vector_data][id_string]
/// Layout for Quantized: [quant_info:1][vector_data][id_string]
///
/// Quantization info byte (only for quantized):
///   - For INT8: value = 0 (no additional info needed)
///   - For PQ8: value = 1
///   - For PQ4: value = 2
///   - For Binary: value = 3
///   - Future: values 4-255 reserved
#[derive(Debug, Clone)]
pub struct CompactVector {
    data: Vec<u8>,
    is_quantized: bool, // Cached for fast access
}

impl CompactVector {
    const QUANT_HEADER_SIZE: usize = 1; // Only for quantized vectors
    const HEADER_SIZE: usize = 1; // Minimum header size

    /// Create from FP32 vector - no header needed!
    pub fn new_fp32(id: &str, vector: &[f32]) -> Result<Self> {
        let id_bytes = id.as_bytes();
        let vector_bytes = bytemuck::cast_slice::<f32, u8>(vector);

        let mut data = Vec::with_capacity(vector_bytes.len() + id_bytes.len());
        data.extend_from_slice(vector_bytes);
        data.extend_from_slice(id_bytes);

        Ok(Self {
            data,
            is_quantized: false,
        })
    }

    /// Create from quantized vector - only 1 byte for quantization method
    pub fn new_quantized(
        id: &str,
        quantized_vector: &[u8],
        quantization_method: u8, // 0=INT8, 1=PQ8, 2=PQ4, 3=Binary
    ) -> Result<Self> {
        let id_bytes = id.as_bytes();

        let mut data = Vec::with_capacity(1 + quantized_vector.len() + id_bytes.len());
        data.push(quantization_method);
        data.extend_from_slice(quantized_vector);
        data.extend_from_slice(id_bytes);

        Ok(Self {
            data,
            is_quantized: true,
        })
    }

    /// Check if quantized
    pub fn is_quantized(&self) -> bool {
        self.is_quantized
    }

    /// Get quantization method (if quantized)
    pub fn quantization_method(&self) -> Option<u8> {
        if self.is_quantized {
            Some(self.data[0])
        } else {
            None
        }
    }

    /// Calculate vector size in bytes - requires external dimension
    fn vector_size_bytes(&self, dimension: usize) -> usize {
        if self.is_quantized {
            // For quantized vectors, calculate based on method and dimension
            let method = self.quantization_method();
            match method {
                Some(0) => dimension,                   // INT8: 1 byte per dimension
                Some(1) => dimension,                   // PQ8: 1 byte per dimension
                Some(2) => (dimension * 4).div_ceil(8), // PQ4: 4 bits per dimension
                Some(3) => dimension.div_ceil(8),       // Binary: 1 bit per dimension
                _ => dimension,                         // Default to 1 byte per dimension
            }
        } else {
            dimension * std::mem::size_of::<f32>()
        }
    }

    /// Get vector as FP32 slice (zero-copy) - requires external dimension
    pub fn vector_as_f32(&self, dimension: usize) -> Result<&[f32]> {
        if self.is_quantized {
            return Err(anyhow!("Vector is quantized, not FP32"));
        }

        let vector_size = dimension * std::mem::size_of::<f32>();
        let bytes = &self.data[..vector_size];
        Ok(bytemuck::cast_slice(bytes))
    }

    /// Get vector as quantized bytes - requires external dimension
    pub fn vector_as_quantized(&self, dimension: usize) -> Result<&[u8]> {
        if !self.is_quantized {
            return Err(anyhow!("Vector is FP32, not quantized"));
        }

        let vector_size = self.vector_size_bytes(dimension);
        Ok(&self.data[Self::QUANT_HEADER_SIZE..Self::QUANT_HEADER_SIZE + vector_size])
    }

    /// Get ID string - requires external dimension
    pub fn id(&self, dimension: usize) -> &str {
        let vector_size = self.vector_size_bytes(dimension);
        let id_start = if self.is_quantized {
            Self::QUANT_HEADER_SIZE + vector_size
        } else {
            vector_size
        };
        let id_bytes = &self.data[id_start..];
        std::str::from_utf8(id_bytes).unwrap_or_default()
    }

    /// Get total size in bytes
    pub fn size_bytes(&self) -> usize {
        self.data.len()
    }

    /// Get raw data (for serialization)
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Create from raw bytes (for deserialization)
    pub fn from_bytes(data: Vec<u8>) -> Result<Self> {
        if data.len() < Self::HEADER_SIZE {
            return Err(anyhow!("Data too small for CompactVector header"));
        }
        Ok(Self {
            data,
            is_quantized: false, // Default to non-quantized
        })
    }
}

/// Collection of compact vectors with dimension stored once
pub struct CompactVectorCollection {
    vectors: Vec<CompactVector>,
    id_index: dashmap::DashMap<String, usize>,
    dimension: usize, // Stored once for the entire collection!
}

impl CompactVectorCollection {
    /// Creates a new compact vector collection with the given vector dimensionality.
    pub fn new(dimension: usize) -> Self {
        Self {
            vectors: Vec::new(),
            id_index: dashmap::DashMap::new(),
            dimension,
        }
    }

    /// Creates a new collection with pre-allocated capacity for the given number of vectors.
    pub fn with_capacity(dimension: usize, capacity: usize) -> Self {
        Self {
            vectors: Vec::with_capacity(capacity),
            id_index: dashmap::DashMap::with_capacity(capacity),
            dimension,
        }
    }

    /// Returns the dimensionality shared by all vectors in this collection.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Adds a full-precision f32 vector with the given ID to the collection.
    pub fn add_fp32(&mut self, id: String, vector: &[f32]) -> Result<()> {
        let compact = CompactVector::new_fp32(&id, vector)?;
        let index = self.vectors.len();
        self.vectors.push(compact);
        self.id_index.insert(id, index);
        Ok(())
    }

    /// Adds a quantized vector with the given ID and quantization method tag.
    pub fn add_quantized(&mut self, id: String, quantized_vector: &[u8], method: u8) -> Result<()> {
        let compact = CompactVector::new_quantized(&id, quantized_vector, method)?;
        let index = self.vectors.len();
        self.vectors.push(compact);
        self.id_index.insert(id, index);
        Ok(())
    }

    /// Looks up a compact vector by its string ID.
    pub fn by_id(&self, id: &str) -> Option<&CompactVector> {
        self.id_index.get(id).map(|index| &self.vectors[*index])
    }

    /// Looks up a compact vector by its positional index in the collection.
    pub fn by_index(&self, index: usize) -> Option<&CompactVector> {
        self.vectors.get(index)
    }

    /// Get vector by ID
    pub fn get_by_id(&self, id: &str) -> Option<&CompactVector> {
        self.id_index.get(id).and_then(|idx| self.vectors.get(*idx))
    }

    /// Get vector as FP32 using stored dimension
    pub fn vector_f32(&self, id: &str) -> Option<Result<&[f32]>> {
        self.get_by_id(id).map(|v| v.vector_as_f32(self.dimension))
    }

    /// Get vector ID using stored dimension
    pub fn vector_id(&self, index: usize) -> Option<&str> {
        self.vectors.get(index).map(|v| v.id(self.dimension))
    }

    /// Returns the number of vectors in the collection.
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// Returns true if the collection contains no vectors.
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Returns the total memory usage in bytes across all stored vectors.
    pub fn memory_usage(&self) -> usize {
        self.vectors.iter().map(|v| v.size_bytes()).sum()
    }

    /// Get vector ID by index (alias for vector_id)
    pub fn get_vector_id(&self, index: usize) -> Option<&str> {
        self.vector_id(index)
    }

    /// Get vector as f32 by ID (alias for vector_f32)
    pub fn get_vector_f32(&self, id: &str) -> Option<Result<&[f32]>> {
        self.vector_f32(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compact_vector_fp32() {
        let id = "test_vector";
        let vector = vec![1.0, 2.0, 3.0, 4.0];
        let dimension = 4;

        let compact = CompactVector::new_fp32(id, &vector).unwrap();

        assert!(!compact.is_quantized());
        assert_eq!(compact.id(dimension), id);

        let retrieved = compact.vector_as_f32(dimension).unwrap();
        assert_eq!(retrieved, &vector[..]);
    }

    #[test]
    fn test_compact_vector_quantized() {
        let id = "quantized_vector";
        let quantized = vec![128, 255, 0, 64]; // 4 bytes for 4D INT8
        let dimension = 4;

        let compact = CompactVector::new_quantized(
            id, &quantized, 0, // INT8
        )
        .unwrap();

        assert!(compact.is_quantized());
        assert_eq!(compact.quantization_method(), Some(0));
        assert_eq!(compact.id(dimension), id);

        let retrieved = compact.vector_as_quantized(dimension).unwrap();
        assert_eq!(retrieved, &quantized[..]);
    }

    #[test]
    fn test_collection() {
        let dimension = 3;
        let mut collection = CompactVectorCollection::new(dimension);

        collection
            .add_fp32("vec1".to_string(), &[1.0, 2.0, 3.0])
            .unwrap();
        collection
            .add_fp32("vec2".to_string(), &[4.0, 5.0, 6.0])
            .unwrap();

        assert_eq!(collection.len(), 2);
        assert_eq!(collection.dimension(), 3);

        let vec1_id = collection.get_vector_id(0).unwrap();
        assert_eq!(vec1_id, "vec1");

        let vec2_data = collection.get_vector_f32("vec2").unwrap().unwrap();
        assert_eq!(vec2_data, &[4.0, 5.0, 6.0]);
    }
}
