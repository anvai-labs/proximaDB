/*
 * Copyright 2025 Vijaykumar Singh
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//! Result Set for Live Query Subscriptions
//!
//! This module provides an efficient data structure for maintaining
//! the top-k results of a live query subscription. It tracks changes
//! and efficiently updates the result set as new vectors are inserted
//! or existing ones are modified.
//!
//! ## Design
//!
//! Uses a BTreeMap keyed by (score, vector_id) for efficient ordered access.
//! This allows O(log n) insertion, removal, and min-score lookup.

use std::collections::{BTreeMap, HashMap};

use serde::{Deserialize, Serialize};

use super::subscription::{ResultChange, ScoredResult};

/// A wrapper around f32 that provides total ordering
/// Used for BTreeMap keys where f32 ordering is needed
use proximadb_records::BitKeyF32 as OrderedFloat;

/// Result set for maintaining top-k results with change tracking
#[derive(Debug)]
pub struct ResultSet {
    /// Results stored by (score, vector_id) for ordered access
    /// Uses OrderedFloat to enable proper ordering of f32 scores
    results: BTreeMap<ResultKey, ScoredResult>,

    /// Index from vector_id to score for fast lookups
    id_index: HashMap<String, OrderedFloat>,

    /// Maximum number of results to keep
    top_k: usize,

    /// Minimum score threshold for inclusion
    score_threshold: f32,
}

/// Key for ordering results in the BTreeMap
/// Orders by score descending (higher scores first), then by vector_id for stability
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct ResultKey {
    /// Negated score for descending order (BTreeMap is ascending)
    neg_score: OrderedFloat,
    /// Vector ID for tie-breaking
    vector_id: String,
}

impl ResultKey {
    fn new(score: f32, vector_id: String) -> Self {
        Self {
            neg_score: OrderedFloat(-score),
            vector_id,
        }
    }

    fn score(&self) -> f32 {
        -self.neg_score.0
    }
}

impl ResultSet {
    /// Create a new result set with the given top-k limit
    pub fn new(top_k: usize) -> Self {
        Self {
            results: BTreeMap::new(),
            id_index: HashMap::new(),
            top_k,
            score_threshold: 0.0,
        }
    }

    /// Create a new result set with top-k limit and score threshold
    pub fn with_threshold(top_k: usize, score_threshold: f32) -> Self {
        Self {
            results: BTreeMap::new(),
            id_index: HashMap::new(),
            top_k,
            score_threshold,
        }
    }

    /// Get the number of results currently in the set
    pub fn len(&self) -> usize {
        self.results.len()
    }

    /// Check if the result set is empty
    pub fn is_empty(&self) -> bool {
        self.results.is_empty()
    }

    /// Get the current minimum score in the result set
    pub fn min_score(&self) -> Option<f32> {
        self.results.keys().last().map(|k| k.score())
    }

    /// Get the current maximum score in the result set
    pub fn max_score(&self) -> Option<f32> {
        self.results.keys().next().map(|k| k.score())
    }

    /// Check if a score would qualify for inclusion
    pub fn would_qualify(&self, score: f32) -> bool {
        if score < self.score_threshold {
            return false;
        }

        if self.results.len() < self.top_k {
            return true;
        }

        // Check against minimum score
        self.min_score().is_none_or(|min| score > min)
    }

    /// Get a result by vector ID
    pub fn get(&self, vector_id: &str) -> Option<&ScoredResult> {
        self.id_index.get(vector_id).and_then(|score| {
            let key = ResultKey::new(score.0, vector_id.to_string());
            self.results.get(&key)
        })
    }

    /// Check if a vector is in the result set
    pub fn contains(&self, vector_id: &str) -> bool {
        self.id_index.contains_key(vector_id)
    }

    /// Get all results as a sorted vector (highest score first)
    pub fn to_vec(&self) -> Vec<ScoredResult> {
        self.results.values().cloned().collect()
    }

    /// Get an iterator over results in score order (highest first)
    pub fn iter(&self) -> impl Iterator<Item = &ScoredResult> {
        self.results.values()
    }

    /// Update the result set with new matches, returns changes
    ///
    /// This is the main method for live query updates. It:
    /// 1. Checks if each new result qualifies for top-k
    /// 2. Updates scores for existing results
    /// 3. Inserts new results that qualify
    /// 4. Removes results that fall out of top-k
    /// 5. Returns all changes that occurred
    pub fn update(&mut self, new_results: Vec<ScoredResult>) -> Vec<ResultChange> {
        let mut changes = Vec::new();

        for result in new_results {
            // Skip if below threshold
            if result.score < self.score_threshold {
                continue;
            }

            // Check if already exists
            if let Some(&existing_score) = self.id_index.get(&result.vector_id) {
                // Score changed?
                if (existing_score.0 - result.score).abs() > f32::EPSILON {
                    // Remove old entry
                    let old_key = ResultKey::new(existing_score.0, result.vector_id.clone());
                    let old_result = self.results.remove(&old_key);
                    let old_position = old_result.as_ref().map_or(0, |r| r.position);

                    // Insert with new score
                    let new_key = ResultKey::new(result.score, result.vector_id.clone());
                    self.results.insert(new_key, result.clone());
                    self.id_index
                        .insert(result.vector_id.clone(), OrderedFloat(result.score));

                    changes.push(ResultChange::ScoreChanged {
                        vector_id: result.vector_id.clone(),
                        old_score: existing_score.0,
                        new_score: result.score,
                        position: old_position, // Will be updated later
                    });
                }
                // Score unchanged - no action needed
            } else {
                // New result - check if it qualifies
                if !self.would_qualify(result.score) {
                    continue;
                }

                // Insert new result
                let key = ResultKey::new(result.score, result.vector_id.clone());
                self.id_index
                    .insert(result.vector_id.clone(), OrderedFloat(result.score));
                self.results.insert(key, result.clone());

                changes.push(ResultChange::Added {
                    vector_id: result.vector_id,
                    score: result.score,
                    position: 0, // Will be updated later
                });
            }
        }

        // Trim to top_k and track removals
        while self.results.len() > self.top_k {
            // Remove the lowest scoring entry (last in BTreeMap)
            if let Some((key, removed)) = self.results.pop_last() {
                self.id_index.remove(&key.vector_id);

                changes.push(ResultChange::Removed {
                    vector_id: removed.vector_id,
                    old_score: removed.score,
                    old_position: removed.position,
                });
            }
        }

        // Update positions for all results
        self.update_positions(&mut changes);

        changes
    }

    /// Remove a vector from the result set
    pub fn remove(&mut self, vector_id: &str) -> Option<ResultChange> {
        if let Some(score) = self.id_index.remove(vector_id) {
            let key = ResultKey::new(score.0, vector_id.to_string());
            if let Some(removed) = self.results.remove(&key) {
                return Some(ResultChange::Removed {
                    vector_id: removed.vector_id,
                    old_score: removed.score,
                    old_position: removed.position,
                });
            }
        }
        None
    }

    /// Remove multiple vectors from the result set
    pub fn remove_many(&mut self, vector_ids: &[String]) -> Vec<ResultChange> {
        let mut changes = Vec::new();

        for vector_id in vector_ids {
            if let Some(change) = self.remove(vector_id) {
                changes.push(change);
            }
        }

        // Update positions after removals
        self.update_positions(&mut changes);

        changes
    }

    /// Clear all results
    pub fn clear(&mut self) -> Vec<ResultChange> {
        let changes: Vec<ResultChange> = self
            .results
            .values()
            .map(|r| ResultChange::Removed {
                vector_id: r.vector_id.clone(),
                old_score: r.score,
                old_position: r.position,
            })
            .collect();

        self.results.clear();
        self.id_index.clear();

        changes
    }

    /// Update positions for all results and changes
    fn update_positions(&mut self, changes: &mut [ResultChange]) {
        // Update positions in the result set
        for (idx, (_, result)) in self.results.iter_mut().enumerate() {
            result.position = idx as u32;
        }

        // Update positions in changes
        for change in changes.iter_mut() {
            match change {
                ResultChange::Added {
                    vector_id,
                    position,
                    ..
                } => {
                    if let Some(result) = self.get(vector_id) {
                        *position = result.position;
                    }
                }
                ResultChange::ScoreChanged {
                    vector_id,
                    position,
                    ..
                } => {
                    if let Some(result) = self.get(vector_id) {
                        *position = result.position;
                    }
                }
                ResultChange::PositionChanged {
                    vector_id,
                    new_position,
                    ..
                } => {
                    if let Some(result) = self.get(vector_id) {
                        *new_position = result.position;
                    }
                }
                ResultChange::Removed { .. } => {
                    // Position stays as the old position
                }
            }
        }
    }

    /// Merge with another result set (used for initial population)
    pub fn merge(&mut self, other: ResultSet) -> Vec<ResultChange> {
        self.update(other.to_vec())
    }

    /// Get the top_k limit
    pub fn top_k(&self) -> usize {
        self.top_k
    }

    /// Get the score threshold
    pub fn score_threshold(&self) -> f32 {
        self.score_threshold
    }
}

impl Default for ResultSet {
    fn default() -> Self {
        Self::new(10)
    }
}

/// Statistics about the result set
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultSetStats {
    /// Number of results
    pub count: usize,
    /// Maximum score
    pub max_score: Option<f32>,
    /// Minimum score
    pub min_score: Option<f32>,
    /// Top-k limit
    pub top_k: usize,
    /// Score threshold
    pub score_threshold: f32,
}

impl From<&ResultSet> for ResultSetStats {
    fn from(rs: &ResultSet) -> Self {
        Self {
            count: rs.len(),
            max_score: rs.max_score(),
            min_score: rs.min_score(),
            top_k: rs.top_k,
            score_threshold: rs.score_threshold,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_result(id: &str, score: f32) -> ScoredResult {
        ScoredResult {
            vector_id: id.to_string(),
            score,
            position: 0,
        }
    }

    #[test]
    fn test_result_set_basic_operations() {
        let mut rs = ResultSet::new(3);

        // Empty initially
        assert!(rs.is_empty());
        assert_eq!(rs.len(), 0);

        // Add some results
        let results = vec![
            create_result("r1", 0.9),
            create_result("r2", 0.8),
            create_result("r3", 0.7),
        ];

        let changes = rs.update(results);
        assert_eq!(changes.len(), 3);
        assert_eq!(rs.len(), 3);
    }

    #[test]
    fn test_result_set_top_k_maintenance() {
        let mut rs = ResultSet::new(3);

        // Add 5 results, should keep top 3
        let results = vec![
            create_result("r1", 0.5),
            create_result("r2", 0.6),
            create_result("r3", 0.7),
            create_result("r4", 0.8),
            create_result("r5", 0.9),
        ];

        let changes = rs.update(results);

        // Should have kept only 3
        assert_eq!(rs.len(), 3);

        // Should have the highest scores
        assert!(rs.contains("r5"));
        assert!(rs.contains("r4"));
        assert!(rs.contains("r3"));
        assert!(!rs.contains("r2"));
        assert!(!rs.contains("r1"));

        // Should have removal changes
        assert!(
            changes
                .iter()
                .any(|c| matches!(c, ResultChange::Removed { .. }))
        );
    }

    #[test]
    fn test_result_set_score_ordering() {
        let mut rs = ResultSet::new(5);

        let results = vec![
            create_result("r3", 0.3),
            create_result("r1", 0.1),
            create_result("r5", 0.5),
            create_result("r2", 0.2),
            create_result("r4", 0.4),
        ];

        rs.update(results);

        // Verify ordering (highest first)
        let vec = rs.to_vec();
        assert_eq!(vec[0].vector_id, "r5");
        assert_eq!(vec[1].vector_id, "r4");
        assert_eq!(vec[2].vector_id, "r3");
        assert_eq!(vec[3].vector_id, "r2");
        assert_eq!(vec[4].vector_id, "r1");

        // Verify positions
        assert_eq!(vec[0].position, 0);
        assert_eq!(vec[1].position, 1);
        assert_eq!(vec[2].position, 2);
    }

    #[test]
    fn test_result_set_score_update() {
        let mut rs = ResultSet::new(3);

        // Initial results
        let results = vec![
            create_result("r1", 0.9),
            create_result("r2", 0.8),
            create_result("r3", 0.7),
        ];
        rs.update(results);

        // Update r3 to have highest score
        let updates = vec![create_result("r3", 0.95)];
        let changes = rs.update(updates);

        // Should have a score change
        assert!(changes.iter().any(|c| matches!(
            c,
            ResultChange::ScoreChanged { vector_id, .. } if vector_id == "r3"
        )));

        // r3 should now be first
        let vec = rs.to_vec();
        assert_eq!(vec[0].vector_id, "r3");
        assert!((vec[0].score - 0.95).abs() < f32::EPSILON);
    }

    #[test]
    fn test_result_set_removal() {
        let mut rs = ResultSet::new(5);

        let results = vec![
            create_result("r1", 0.9),
            create_result("r2", 0.8),
            create_result("r3", 0.7),
        ];
        rs.update(results);

        // Remove r2
        let change = rs.remove("r2");
        assert!(change.is_some());
        assert_eq!(rs.len(), 2);
        assert!(!rs.contains("r2"));
    }

    #[test]
    fn test_result_set_threshold() {
        let mut rs = ResultSet::with_threshold(5, 0.5);

        let results = vec![
            create_result("r1", 0.9),
            create_result("r2", 0.4), // Below threshold
            create_result("r3", 0.6),
            create_result("r4", 0.3), // Below threshold
        ];

        rs.update(results);

        // Should only include above threshold
        assert_eq!(rs.len(), 2);
        assert!(rs.contains("r1"));
        assert!(rs.contains("r3"));
        assert!(!rs.contains("r2"));
        assert!(!rs.contains("r4"));
    }

    #[test]
    fn test_result_set_would_qualify() {
        let mut rs = ResultSet::new(3);

        // Empty set - anything qualifies
        assert!(rs.would_qualify(0.1));

        // Fill up
        let results = vec![
            create_result("r1", 0.9),
            create_result("r2", 0.8),
            create_result("r3", 0.7),
        ];
        rs.update(results);

        // Higher than min (0.7) qualifies
        assert!(rs.would_qualify(0.8));
        assert!(rs.would_qualify(0.95));

        // Lower than min doesn't qualify
        assert!(!rs.would_qualify(0.6));
        assert!(!rs.would_qualify(0.1));
    }

    #[test]
    fn test_result_set_min_max_score() {
        let mut rs = ResultSet::new(5);

        assert!(rs.min_score().is_none());
        assert!(rs.max_score().is_none());

        let results = vec![
            create_result("r1", 0.9),
            create_result("r2", 0.5),
            create_result("r3", 0.7),
        ];
        rs.update(results);

        assert!((rs.max_score().unwrap() - 0.9).abs() < f32::EPSILON);
        assert!((rs.min_score().unwrap() - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_result_set_clear() {
        let mut rs = ResultSet::new(5);

        let results = vec![create_result("r1", 0.9), create_result("r2", 0.8)];
        rs.update(results);

        let changes = rs.clear();
        assert_eq!(changes.len(), 2);
        assert!(rs.is_empty());
    }

    #[test]
    fn test_result_set_remove_many() {
        let mut rs = ResultSet::new(5);

        let results = vec![
            create_result("r1", 0.9),
            create_result("r2", 0.8),
            create_result("r3", 0.7),
            create_result("r4", 0.6),
        ];
        rs.update(results);

        let changes = rs.remove_many(&["r1".to_string(), "r3".to_string()]);
        assert_eq!(changes.len(), 2);
        assert_eq!(rs.len(), 2);
        assert!(!rs.contains("r1"));
        assert!(!rs.contains("r3"));
    }

    #[test]
    fn test_result_set_get() {
        let mut rs = ResultSet::new(5);

        let results = vec![create_result("r1", 0.9)];
        rs.update(results);

        let result = rs.get("r1");
        assert!(result.is_some());
        assert_eq!(result.unwrap().vector_id, "r1");

        assert!(rs.get("nonexistent").is_none());
    }

    #[test]
    fn test_result_key_ordering() {
        let k1 = ResultKey::new(0.9, "a".to_string());
        let k2 = ResultKey::new(0.8, "a".to_string());
        let k3 = ResultKey::new(0.9, "b".to_string());

        // Higher score should come first (smaller key)
        assert!(k1 < k2);

        // Same score, alphabetical order
        assert!(k1 < k3);
    }
}
