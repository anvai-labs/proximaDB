//! Memory-mapped file support for large vector datasets
//!
//! This module provides efficient memory-mapped I/O for vector storage,
//! allowing direct access to large datasets without loading them into memory.

use anyhow::{Context, Result};
use memmap2::{Mmap, MmapMut, MmapOptions};
use parking_lot::RwLock;
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Configuration for memory-mapped vector storage
#[derive(Debug, Clone)]
pub struct MmapConfig {
    /// Enable memory-mapped files for vectors
    pub enabled: bool,
    /// Minimum file size to use mmap (bytes)
    pub min_size: usize,
    /// Use huge pages if available
    pub use_huge_pages: bool,
    /// Populate pages on mmap (prefault)
    pub populate: bool,
    /// Lock pages in memory (requires privileges)
    pub lock_memory: bool,
}

impl Default for MmapConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_size: 1024 * 1024, // 1MB minimum
            use_huge_pages: false,
            populate: false,
            lock_memory: false,
        }
    }
}

/// Memory-mapped vector storage
pub struct MmapVectorStorage {
    config: MmapConfig,
    mappings: Arc<RwLock<Vec<MmapHandle>>>,
}

struct MmapHandle {
    #[allow(dead_code)]
    path: PathBuf,
    #[allow(dead_code)]
    mmap: Arc<Mmap>,
    #[allow(dead_code)]
    size: usize,
    #[allow(dead_code)]
    vector_count: usize,
    #[allow(dead_code)]
    dimension: usize,
}

impl MmapVectorStorage {
    /// Create new memory-mapped vector storage
    pub fn new(config: MmapConfig) -> Self {
        Self {
            config,
            mappings: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Map a vector file into memory
    pub fn map_file(&self, path: impl AsRef<Path>, dimension: usize) -> Result<Arc<Mmap>> {
        let path = path.as_ref();
        let file =
            File::open(path).with_context(|| format!("Failed to open vector file: {:?}", path))?;

        let metadata = file.metadata()?;
        let file_size = metadata.len() as usize;

        // Check if file is large enough for mmap
        if !self.config.enabled || file_size < self.config.min_size {
            // Fall back to regular I/O for small files
            return Err(anyhow::anyhow!(
                "File too small for mmap: {} bytes",
                file_size
            ));
        }

        // Create memory mapping
        let mut mmap_opts = MmapOptions::new();

        if self.config.populate {
            mmap_opts.populate();
        }

        let mmap = unsafe { mmap_opts.map(&file)? };

        // Lock pages if requested (requires privileges)
        if self.config.lock_memory {
            #[cfg(unix)]
            {
                use libc::{_SC_PAGESIZE, mlock, sysconf};
                let page_size = unsafe { sysconf(_SC_PAGESIZE) } as usize;
                let aligned_size = (file_size + page_size - 1) & !(page_size - 1);

                unsafe {
                    if mlock(mmap.as_ptr() as *const _, aligned_size) != 0 {
                        tracing::warn!("Failed to lock memory pages (requires privileges)");
                    }
                }
            }
        }

        // Calculate vector count (assuming f32 vectors)
        let bytes_per_vector = dimension * std::mem::size_of::<f32>();
        let vector_count = file_size / bytes_per_vector;

        let mmap_arc = Arc::new(mmap);

        let handle = MmapHandle {
            path: path.to_path_buf(),
            mmap: mmap_arc.clone(),
            size: file_size,
            vector_count,
            dimension,
        };

        self.mappings.write().push(handle);

        Ok(mmap_arc)
    }

    /// Create a new memory-mapped file for writing vectors
    pub fn create_mmap_file(
        &self,
        path: impl AsRef<Path>,
        vector_count: usize,
        dimension: usize,
    ) -> Result<MmapMut> {
        let path = path.as_ref();
        let bytes_per_vector = dimension * std::mem::size_of::<f32>();
        let file_size = vector_count * bytes_per_vector;

        // Create and size the file
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;

        file.set_len(file_size as u64)?;

        // Create mutable memory mapping
        let mmap = unsafe { MmapMut::map_mut(&file)? };

        Ok(mmap)
    }

    /// Read vectors directly from memory-mapped file
    pub fn read_vectors(
        &self,
        mmap: &Mmap,
        start_idx: usize,
        count: usize,
        dimension: usize,
    ) -> Result<Vec<Vec<f32>>> {
        let bytes_per_vector = dimension * std::mem::size_of::<f32>();
        let start_byte = start_idx * bytes_per_vector;
        let end_byte = (start_idx + count) * bytes_per_vector;

        if end_byte > mmap.len() {
            return Err(anyhow::anyhow!("Read exceeds mmap bounds"));
        }

        let mut vectors = Vec::with_capacity(count);
        let data = &mmap[start_byte..end_byte];

        for i in 0..count {
            let offset = i * bytes_per_vector;
            let vector_bytes = &data[offset..offset + bytes_per_vector];

            // Convert bytes to f32 vector
            let vector: Vec<f32> = vector_bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|chunk| f32::from_le_bytes(*chunk))
                .collect();

            vectors.push(vector);
        }

        Ok(vectors)
    }

    /// Get a zero-copy slice of vectors
    ///
    /// # Safety
    ///
    /// Callers must ensure:
    /// - `start_idx + count` does not exceed the number of vectors in the mmap
    /// - The mmap contains valid vector data for the requested range
    /// - `dimension` matches the actual dimension of vectors in the mmap
    pub unsafe fn get_vector_slice(
        mmap: &Mmap,
        start_idx: usize,
        count: usize,
        dimension: usize,
    ) -> &[f32] {
        unsafe {
            let floats_per_vector = dimension;
            let start_float = start_idx * floats_per_vector;
            let total_floats = count * floats_per_vector;

            let ptr = mmap.as_ptr() as *const f32;
            std::slice::from_raw_parts(ptr.add(start_float), total_floats)
        }
    }

    /// Prefetch vectors into CPU cache
    #[cfg(target_arch = "x86_64")]
    pub fn prefetch_vectors(&self, mmap: &Mmap, start_idx: usize, count: usize, dimension: usize) {
        use std::arch::x86_64::_mm_prefetch;

        let bytes_per_vector = dimension * std::mem::size_of::<f32>();
        let start_byte = start_idx * bytes_per_vector;

        unsafe {
            for i in 0..count {
                let offset = start_byte + i * bytes_per_vector;
                if offset < mmap.len() {
                    _mm_prefetch(mmap.as_ptr().add(offset) as *const i8, 3); // T2 hint
                }
            }
        }
    }

    /// Get statistics about memory-mapped files
    pub fn stats(&self) -> MmapStats {
        let mappings = self.mappings.read();

        MmapStats {
            total_files: mappings.len(),
            total_size: mappings.iter().map(|h| h.size).sum(),
            total_vectors: mappings.iter().map(|h| h.vector_count).sum(),
        }
    }
}

/// Statistics for memory-mapped storage
#[derive(Debug, Clone)]
pub struct MmapStats {
    pub total_files: usize,
    pub total_size: usize,
    pub total_vectors: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_mmap_vector_storage() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let path = temp_dir.path().join("vectors.bin");

        // Use custom config with lower min_size for testing
        let config = MmapConfig {
            enabled: true,
            min_size: 100_000, // 100KB - lower threshold for testing
            use_huge_pages: false,
            populate: false,
            lock_memory: false,
        };
        let storage = MmapVectorStorage::new(config);

        // Create a file with some vectors
        let dimension = 128;
        let vector_count = 1000;
        let mut mmap_mut = storage.create_mmap_file(&path, vector_count, dimension)?;

        // Write test data
        let test_data: Vec<f32> = (0..dimension * vector_count)
            .map(|i| i as f32 / 100.0)
            .collect();

        for (i, chunk) in test_data.chunks_exact(dimension).enumerate() {
            let offset = i * dimension * 4;
            for (j, &val) in chunk.iter().enumerate() {
                let bytes = val.to_le_bytes();
                mmap_mut[offset + j * 4..offset + j * 4 + 4].copy_from_slice(&bytes);
            }
        }

        drop(mmap_mut);

        // Map the file and read vectors
        let mmap = storage.map_file(&path, dimension)?;
        let vectors = storage.read_vectors(&mmap, 0, 10, dimension)?;

        assert_eq!(vectors.len(), 10);
        assert_eq!(vectors[0].len(), dimension);

        Ok(())
    }
}
