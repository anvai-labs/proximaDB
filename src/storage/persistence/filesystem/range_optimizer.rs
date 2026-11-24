//! Range Optimizer for Unified Filesystem
//!
//! Optimizes range-based reads for large files, particularly beneficial
//! for cloud storage where partial reads can significantly reduce bandwidth.

use std::cmp;
use std::sync::Arc;

use dashmap::DashMap;
use tracing::{debug, trace};

/// Range optimizer for intelligent partial file reads
pub struct RangeOptimizer {
    /// Threshold for merging adjacent ranges (bytes)
    merge_threshold: usize,

    /// Minimum file size for range optimization (bytes)
    min_file_size: usize,

    /// Range history for pattern learning
    range_history: Arc<DashMap<String, Vec<AccessedRange>>>,

    /// Statistics
    stats: Arc<RangeStats>,
}

/// Represents an accessed range in a file
#[derive(Debug, Clone)]
struct AccessedRange {
    start: u64,
    end: u64,
    access_count: u32,
}

/// Optimized range for reading
#[derive(Debug, Clone)]
pub struct OptimizedRange {
    pub start: u64,
    pub end: u64,
}

/// Range optimization statistics
#[derive(Debug, Default)]
struct RangeStats {
    optimizations: std::sync::atomic::AtomicU64,
    bytes_saved: std::sync::atomic::AtomicU64,
    ranges_merged: std::sync::atomic::AtomicU64,
}

impl RangeOptimizer {
    /// Create new range optimizer
    pub fn new(merge_threshold: usize, min_file_size_mb: usize) -> Self {
        Self {
            merge_threshold,
            min_file_size: min_file_size_mb * 1024 * 1024,
            range_history: Arc::new(DashMap::new()),
            stats: Arc::new(RangeStats::default()),
        }
    }

    /// Optimize ranges for reading based on access patterns
    pub async fn optimize_ranges(&self, file_path: &str, file_size: u64) -> Vec<OptimizedRange> {
        // Check if file is large enough for range optimization
        if file_size < self.min_file_size as u64 {
            return vec![];
        }

        // Get historical access patterns
        if let Some(history) = self.range_history.get(file_path) {
            let ranges = self.analyze_patterns(&history);
            if !ranges.is_empty() {
                self.stats
                    .optimizations
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return self.merge_ranges(ranges);
            }
        }

        // No historical patterns, return empty (full file read)
        vec![]
    }

    /// Determine optimal range strategy based on storage engine
    pub async fn optimize_engine_ranges(
        &self,
        file_path: &str,
        file_size: u64,
        storage_engine: &str,
        _column_indices: Option<Vec<usize>>,
        row_group_indices: Option<Vec<usize>>,
    ) -> Vec<OptimizedRange> {
        let mut ranges = Vec::new();

        // Optimize based on storage engine type
        match storage_engine.to_lowercase().as_str() {
            "viper" | "nova" => {
                // NOVA uses Parquet format
                // Always need the footer (last 8 bytes + footer size)
                // Estimate footer size as 1% of file or 64KB, whichever is smaller
                let footer_size = std::cmp::min(file_size / 100, 65536);
                ranges.push(OptimizedRange {
                    start: file_size.saturating_sub(footer_size),
                    end: file_size,
                });

                // If specific row groups requested, calculate their ranges
                if let Some(row_groups) = row_group_indices {
                    // Estimate row group size (divide file by typical number of row groups)
                    let estimated_rg_size = file_size / 10; // Assume ~10 row groups
                    for rg_idx in row_groups {
                        let start = (rg_idx as u64) * estimated_rg_size;
                        let end = std::cmp::min(start + estimated_rg_size, file_size);
                        ranges.push(OptimizedRange { start, end });
                    }
                }

                // If specific columns requested, we'd need the footer first to determine column chunks
                // For now, this is a placeholder for column-specific optimization

                self.merge_ranges(ranges)
            }
            "sst" | "lsm" => {
                // SST uses sorted string tables, typically need header and index blocks
                // Read first 4KB for header and last portion for index
                ranges.push(OptimizedRange {
                    start: 0,
                    end: std::cmp::min(4096, file_size),
                });

                // Index typically at the end
                let index_size = std::cmp::min(file_size / 20, 32768);
                ranges.push(OptimizedRange {
                    start: file_size.saturating_sub(index_size),
                    end: file_size,
                });

                self.merge_ranges(ranges)
            }
            "swift" => {
                // SWIFT uses Proxima encoding with superblocks
                // Need header and potentially specific superblocks
                ranges.push(OptimizedRange {
                    start: 0,
                    end: std::cmp::min(8192, file_size), // Superblock header
                });

                // Add ranges for requested superblocks if specified
                self.merge_ranges(ranges)
            }
            "raptor" => {
                // RAPTOR uses adaptive row groups
                // Similar to Parquet but with different chunking
                let footer_size = std::cmp::min(file_size / 50, 32768);
                ranges.push(OptimizedRange {
                    start: file_size.saturating_sub(footer_size),
                    end: file_size,
                });

                self.merge_ranges(ranges)
            }
            _ => {
                // Unknown engine or engines that don't benefit from range optimization
                // Use regular optimization based on access patterns
                self.optimize_ranges(file_path, file_size).await
            }
        }
    }

    /// Record a range access for learning
    pub async fn record_access(&self, file_path: &str, start: u64, end: u64) {
        let mut entry = self.range_history.entry(file_path.to_string()).or_default();

        // Check if this range already exists
        for range in entry.iter_mut() {
            if range.overlaps(start, end) {
                range.merge(start, end);
                return;
            }
        }

        // Add new range
        entry.push(AccessedRange {
            start,
            end,
            access_count: 1,
        });

        // Keep history limited to prevent memory growth
        if entry.len() > 100 {
            entry.drain(0..50);
        }
    }

    /// Analyze access patterns to predict useful ranges
    fn analyze_patterns(&self, history: &[AccessedRange]) -> Vec<OptimizedRange> {
        // Find frequently accessed ranges (access_count > 2)
        let frequent_ranges: Vec<OptimizedRange> = history
            .iter()
            .filter(|r| r.access_count > 2)
            .map(|r| OptimizedRange {
                start: r.start,
                end: r.end,
            })
            .collect();

        trace!("Found {} frequently accessed ranges", frequent_ranges.len());
        frequent_ranges
    }

    /// Merge adjacent or overlapping ranges
    fn merge_ranges(&self, mut ranges: Vec<OptimizedRange>) -> Vec<OptimizedRange> {
        if ranges.is_empty() {
            return ranges;
        }

        // Sort by start position
        ranges.sort_by_key(|r| r.start);

        let mut merged = Vec::new();
        let mut current = ranges[0].clone();

        for range in ranges.iter().skip(1) {
            if range.start <= current.end + self.merge_threshold as u64 {
                // Merge ranges
                current.end = cmp::max(current.end, range.end);
                self.stats
                    .ranges_merged
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            } else {
                // Gap too large, start new range
                merged.push(current);
                current = range.clone();
            }
        }

        merged.push(current);

        debug!("Merged ranges: {} -> {}", ranges.len(), merged.len());
        merged
    }

    /// Calculate potential bandwidth savings
    pub async fn calculate_savings(&self, file_size: u64, ranges: &[OptimizedRange]) -> u64 {
        if ranges.is_empty() {
            return 0;
        }

        let bytes_to_read: u64 = ranges.iter().map(|r| r.end - r.start).sum();
        let saved = file_size.saturating_sub(bytes_to_read);

        self.stats
            .bytes_saved
            .fetch_add(saved, std::sync::atomic::Ordering::Relaxed);
        saved
    }

    /// Get optimization statistics
    pub fn stats(&self) -> RangeOptimizationStats {
        RangeOptimizationStats {
            optimizations: self
                .stats
                .optimizations
                .load(std::sync::atomic::Ordering::Relaxed),
            bytes_saved: self
                .stats
                .bytes_saved
                .load(std::sync::atomic::Ordering::Relaxed),
            ranges_merged: self
                .stats
                .ranges_merged
                .load(std::sync::atomic::Ordering::Relaxed),
        }
    }
}

impl AccessedRange {
    /// Check if this range overlaps with another
    fn overlaps(&self, start: u64, end: u64) -> bool {
        self.start <= end && start <= self.end
    }

    /// Merge with another range
    fn merge(&mut self, start: u64, end: u64) {
        self.start = cmp::min(self.start, start);
        self.end = cmp::max(self.end, end);
        self.access_count += 1;
    }
}

/// Public range optimization statistics
#[derive(Debug, Clone)]
pub struct RangeOptimizationStats {
    pub optimizations: u64,
    pub bytes_saved: u64,
    pub ranges_merged: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_range_merging() {
        let optimizer = RangeOptimizer::new(50, 1);

        let ranges = vec![
            OptimizedRange { start: 0, end: 100 },
            OptimizedRange {
                start: 110,
                end: 200,
            },
            OptimizedRange {
                start: 1100,
                end: 1200,
            },
            OptimizedRange {
                start: 1150,
                end: 1300,
            },
        ];

        let merged = optimizer.merge_ranges(ranges);

        // Should merge first two (gap < threshold) and last two (overlapping)
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].start, 0);
        assert_eq!(merged[0].end, 200);
        assert_eq!(merged[1].start, 1100);
        assert_eq!(merged[1].end, 1300);
    }

    #[tokio::test]
    async fn test_access_recording() {
        let optimizer = RangeOptimizer::new(1024, 1);

        // Record multiple accesses
        optimizer.record_access("test.parquet", 0, 100).await;
        optimizer.record_access("test.parquet", 50, 150).await;
        optimizer.record_access("test.parquet", 1000, 2000).await;

        let history = optimizer.range_history.get("test.parquet");
        assert!(history.is_some());

        let ranges = history.unwrap();
        assert_eq!(ranges.len(), 2); // First two should merge
    }
}
