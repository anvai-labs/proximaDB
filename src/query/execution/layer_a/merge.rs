// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0
//! The newest-wins k-way merge over oid-sorted segment streams.
//!
//! Every spill segment holds at most one row per oid, sorted by oid (the flush
//! writes it so and declares it). The merge walks all segments in step, oldest to
//! newest: for each oid the copy in the NEWEST segment holding it wins, and older
//! copies are consumed silently. A winner is then dropped if the partition's
//! memory layer shadows it — a resident row (strictly newer than any segment
//! copy) or a memory tombstone — and otherwise emitted if visible.
//!
//! Visibility is decided only AFTER resolution, so a dead or foreign newest copy
//! hides every older one rather than letting it through.
//!
//! Resident rows are emitted last, minus tombstoned oids (a tombstone wins over a
//! resident copy, as every point read serves it).
//!
//! **Memory:** one conformed batch per segment plus the rows pending output —
//! `O(k × batch)`, independent of table size.
//!
//! **Order is verified, not assumed.** A segment whose oids do not strictly
//! increase fails the scan: merging it would silently serve stale copies.

use std::collections::HashSet;

use anyhow::{Result, anyhow};
use arrow::array::{Array, ArrayRef};
use arrow::compute::{filter_record_batch, interleave};
use arrow::datatypes::SchemaRef;
use arrow::record_batch::{RecordBatch, RecordBatchOptions};
use futures::StreamExt;
use futures::stream::BoxStream;

use super::conform::Conformed;

pub(crate) type ConformedStream = BoxStream<'static, Result<Conformed>>;

struct Cursor {
    stream: Option<ConformedStream>,
    /// Index into `LayerAMerge::slots` of the batch being walked.
    slot: Option<usize>,
    row: usize,
    /// The last oid of the previous batch, to verify order across batches.
    last_oid: Option<String>,
}

pub(crate) struct LayerAMerge {
    /// Oldest → newest.
    cursors: Vec<Cursor>,
    /// Oids a segment copy must not be served for: resident and tombstoned.
    shadow: HashSet<String>,
    /// Emitted after the segments, once.
    resident: Option<Conformed>,
    output: SchemaRef,
    batch_size: usize,
    /// Rows still to emit under the scan's limit.
    remaining: Option<usize>,
    /// Batches a pending row or a cursor still refers to.
    slots: Vec<Conformed>,
    /// `(slot, row)` of each row of the next output batch.
    pending: Vec<(usize, usize)>,
}

impl LayerAMerge {
    pub fn new(
        segments: Vec<ConformedStream>,
        shadow: HashSet<String>,
        resident: Option<Conformed>,
        output: SchemaRef,
        batch_size: usize,
        limit: Option<usize>,
    ) -> Self {
        Self {
            cursors: segments
                .into_iter()
                .map(|stream| Cursor {
                    stream: Some(stream),
                    slot: None,
                    row: 0,
                    last_oid: None,
                })
                .collect(),
            shadow,
            resident,
            output,
            batch_size: batch_size.max(1),
            remaining: limit,
            slots: Vec::new(),
            pending: Vec::new(),
        }
    }

    fn oid_at(&self, (slot, row): (usize, usize)) -> &str {
        self.slots[slot].oids.value(row)
    }

    /// Give cursor `i` a row to look at, pulling batches until one is non-empty
    /// or its stream ends.
    async fn refill(&mut self, i: usize) -> Result<()> {
        loop {
            let c = &self.cursors[i];
            if let Some(slot) = c.slot {
                if c.row < self.slots[slot].len() {
                    return Ok(());
                }
                let last = self.slots[slot].oids.value(c.row - 1).to_string();
                let c = &mut self.cursors[i];
                c.last_oid = Some(last);
                c.slot = None;
            }
            let Some(stream) = self.cursors[i].stream.as_mut() else {
                return Ok(());
            };
            match stream.next().await {
                None => {
                    self.cursors[i].stream = None;
                    return Ok(());
                }
                Some(batch) => {
                    let batch = batch?;
                    if batch.len() == 0 {
                        continue;
                    }
                    if let Some(last) = &self.cursors[i].last_oid
                        && batch.oids.value(0) <= last.as_str()
                    {
                        return Err(out_of_order(i, last, batch.oids.value(0)));
                    }
                    self.slots.push(batch);
                    let c = &mut self.cursors[i];
                    c.slot = Some(self.slots.len() - 1);
                    c.row = 0;
                }
            }
        }
    }

    /// Step cursor `i`, at `(slot, row)`, past that row, verifying the next one
    /// sorts after.
    fn advance(&mut self, i: usize, (slot, row): (usize, usize)) -> Result<()> {
        let oids = &self.slots[slot].oids;
        let next = row + 1;
        if next < oids.len() && oids.value(next) <= oids.value(row) {
            return Err(out_of_order(i, oids.value(row), oids.value(next)));
        }
        self.cursors[i].row = next;
        Ok(())
    }

    /// The next output batch, or `None` when the scan is complete.
    pub async fn next_batch(&mut self) -> Result<Option<RecordBatch>> {
        loop {
            if self.remaining == Some(0) {
                return self.flush();
            }
            for i in 0..self.cursors.len() {
                self.refill(i).await?;
            }
            // Batches walked past with nothing served (superseded copies, dead
            // or filtered rows) would otherwise stay referenced until the next
            // output batch — the whole table, when few rows are served.
            // With rows pending, emit them early instead: a few sparse rows
            // must not pin a batch each. Either way at most ~2k batches stay
            // referenced — memory is O(segments × batch).
            if self.slots.len() > 2 * self.cursors.len() {
                if self.pending.is_empty() {
                    self.release_unreferenced()?;
                } else {
                    return self.flush();
                }
            }
            // `(cursor, (slot, row))` of every cursor still holding a row.
            let active: Vec<(usize, (usize, usize))> = self
                .cursors
                .iter()
                .enumerate()
                .filter_map(|(i, c)| c.slot.map(|slot| (i, (slot, c.row))))
                .collect();
            let Some(&first) = active.first() else {
                if !self.pending.is_empty() {
                    return self.flush();
                }
                return self.emit_resident();
            };

            // The smallest current oid; among equals the newest segment wins
            // (cursors are ordered oldest → newest, so the last equal one).
            let mut winner = first;
            for &candidate in &active[1..] {
                if self.oid_at(candidate.1) <= self.oid_at(winner.1) {
                    winner = candidate;
                }
            }
            let (slot, row) = winner.1;
            let serve = self.slots[slot].visible.value(row)
                && (self.shadow.is_empty() || !self.shadow.contains(self.oid_at(winner.1)));
            let same_oid: Vec<(usize, (usize, usize))> = active
                .iter()
                .copied()
                .filter(|&(_, at)| self.oid_at(at) == self.oid_at(winner.1))
                .collect();
            for (i, at) in same_oid {
                self.advance(i, at)?;
            }
            if serve {
                self.pending.push((slot, row));
                if let Some(remaining) = self.remaining.as_mut() {
                    *remaining -= 1;
                }
                if self.pending.len() >= self.batch_size {
                    return self.flush();
                }
            }
        }
    }

    /// Materialize the pending rows, then drop every batch no cursor still walks.
    fn flush(&mut self) -> Result<Option<RecordBatch>> {
        if self.pending.is_empty() {
            return Ok(None);
        }
        let n = self.pending.len();
        let columns = (0..self.output.fields().len())
            .map(|col| {
                let arrays: Vec<&dyn Array> = self
                    .slots
                    .iter()
                    .map(|s| s.batch.column(col).as_ref())
                    .collect();
                interleave(&arrays, &self.pending)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let batch = self.emit(columns, n)?;
        self.pending.clear();
        self.release_unreferenced()?;
        Ok(Some(batch))
    }

    /// Drop every batch no cursor is walking. Only valid with nothing pending
    /// (pending rows refer to slots by index).
    fn release_unreferenced(&mut self) -> Result<()> {
        let mut old: Vec<Option<Conformed>> = std::mem::take(&mut self.slots)
            .into_iter()
            .map(Some)
            .collect();
        for c in &mut self.cursors {
            if let Some(slot) = c.slot {
                let batch = old[slot].take().ok_or_else(|| {
                    anyhow!("layer A: merge slot {slot} is walked by two cursors")
                })?;
                self.slots.push(batch);
                c.slot = Some(self.slots.len() - 1);
            }
        }
        Ok(())
    }

    /// Rows that won the merge and are visible, under the scan's declared
    /// schema (conformed batches use an all-nullable one; see `ConformContext`).
    fn emit(&self, columns: Vec<ArrayRef>, rows: usize) -> Result<RecordBatch> {
        let options = RecordBatchOptions::new().with_row_count(Some(rows));
        Ok(RecordBatch::try_new_with_options(
            self.output.clone(),
            columns,
            &options,
        )?)
    }

    fn emit_resident(&mut self) -> Result<Option<RecordBatch>> {
        let Some(resident) = self.resident.take() else {
            return Ok(None);
        };
        let mut batch = filter_record_batch(&resident.batch, &resident.visible)?;
        if let Some(remaining) = self.remaining.as_mut() {
            batch = batch.slice(0, batch.num_rows().min(*remaining));
            *remaining -= batch.num_rows();
        }
        if batch.num_rows() == 0 {
            return Ok(None);
        }
        let n = batch.num_rows();
        self.emit(batch.columns().to_vec(), n).map(Some)
    }
}

fn out_of_order(segment: usize, before: &str, after: &str) -> anyhow::Error {
    anyhow!(
        "layer A: segment #{segment} is not strictly ordered by oid ('{before}' then \
         '{after}'); refusing to merge it rather than serve a stale copy"
    )
}
