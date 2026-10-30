//! Bounded-memory, oid-ordered record scans (P2.4).
//!
//! A [`RecordChunkStream`] yields a store's live records in chunks, in **strictly
//! ascending `oid` order** (byte-wise `String` order) with each oid at most once.
//! The ordering is the contract that lets independent sources combine without
//! materializing either: [`merge_newest_wins`] k-way merges sorted sources (spill
//! segments under a memtable, a primary partition over a legacy one) holding one
//! chunk per source.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, VecDeque};

use futures::StreamExt;
use futures::stream::{self, BoxStream};

use crate::ProximaRecord;
use crate::store::RecordStoreResult;

/// Live records in ascending, unique `oid` order, in chunks of at most the
/// requested size (never empty).
pub type RecordChunkStream<'a> = BoxStream<'a, RecordStoreResult<Vec<ProximaRecord>>>;

/// A chunk size of zero is treated as one, so a caller typo cannot stall a scan.
pub fn effective_chunk_rows(chunk_rows: usize) -> usize {
    chunk_rows.max(1)
}

/// Serve an already-materialized record set as a chunk stream: sorts by oid and
/// keeps the last record of a duplicated oid. The fallback for stores that have
/// no ordered scan; its memory is the whole set.
pub fn chunks_of_records<'a>(
    mut records: Vec<ProximaRecord>,
    chunk_rows: usize,
) -> RecordChunkStream<'a> {
    let chunk_rows = effective_chunk_rows(chunk_rows);
    // Stable sort, so "last" is the input's last copy of the oid.
    records.sort_by(|a, b| a.oid.cmp(&b.oid));
    let mut deduped: Vec<ProximaRecord> = Vec::with_capacity(records.len());
    for record in records {
        match deduped.last_mut() {
            Some(last) if last.oid == record.oid => *last = record,
            _ => deduped.push(record),
        }
    }
    let mut rest = deduped.into_iter();
    stream::iter(std::iter::from_fn(move || {
        let chunk: Vec<ProximaRecord> = rest.by_ref().take(chunk_rows).collect();
        (!chunk.is_empty()).then_some(Ok(chunk))
    }))
    .boxed()
}

/// Run `scan` when the stream is first polled and serve its result as chunks
/// ([`chunks_of_records`]); its error is the stream's only item.
pub fn once_then_chunks<'a, F>(scan: F, chunk_rows: usize) -> RecordChunkStream<'a>
where
    F: std::future::Future<Output = RecordStoreResult<Vec<ProximaRecord>>> + Send + 'a,
{
    flatten_setup(async move {
        scan.await
            .map(|records| chunks_of_records(records, chunk_rows))
    })
}

/// Run `setup` (which builds the real stream) when first polled; a setup error
/// is the stream's only item.
pub fn flatten_setup<'a, F>(setup: F) -> RecordChunkStream<'a>
where
    F: std::future::Future<Output = RecordStoreResult<RecordChunkStream<'a>>> + Send + 'a,
{
    stream::once(setup)
        .map(|setup| match setup {
            Ok(stream) => stream,
            Err(e) => stream::once(async move { Err(e) }).boxed(),
        })
        .flatten()
        .boxed()
}

/// Serve a key-addressed store by a snapshot of its keys: `keys` is sorted and
/// deduplicated here, and each chunk's records are fetched when the chunk is
/// produced. A key whose record has gone (deleted since the snapshot) or that
/// `keep` rejects is skipped. Memory is the key set plus one chunk of records.
pub fn chunks_by_key_snapshot<'a, F>(
    mut keys: Vec<String>,
    chunk_rows: usize,
    fetch: F,
) -> RecordChunkStream<'a>
where
    F: Fn(&str) -> Option<ProximaRecord> + Send + 'a,
{
    let chunk_rows = effective_chunk_rows(chunk_rows);
    keys.sort_unstable();
    keys.dedup();
    let mut keys = keys.into_iter();
    stream::iter(std::iter::from_fn(move || {
        let mut chunk = Vec::new();
        for key in keys.by_ref() {
            if let Some(record) = fetch(&key) {
                chunk.push(record);
                if chunk.len() >= chunk_rows {
                    break;
                }
            }
        }
        (!chunk.is_empty()).then_some(Ok(chunk))
    }))
    .boxed()
}

/// Keep only the records `keep` accepts, dropping chunks that become empty.
pub fn filter_chunks<'a, F>(chunks: RecordChunkStream<'a>, keep: F) -> RecordChunkStream<'a>
where
    F: Fn(&ProximaRecord) -> bool + Send + 'a,
{
    chunks
        .filter_map(move |chunk| {
            let out = match chunk {
                Ok(mut records) => {
                    records.retain(|r| keep(r));
                    (!records.is_empty()).then_some(Ok(records))
                }
                Err(e) => Some(Err(e)),
            };
            futures::future::ready(out)
        })
        .boxed()
}

/// One buffered record in the merge heap. Ordered so the max-heap pops the
/// smallest oid first and, among equal oids, the newest source first.
struct Head {
    record: ProximaRecord,
    source: usize,
}

impl PartialEq for Head {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Head {}
impl PartialOrd for Head {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Head {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .record
            .oid
            .cmp(&self.record.oid)
            .then(self.source.cmp(&other.source))
    }
}

struct Source<'a> {
    stream: RecordChunkStream<'a>,
    buffered: VecDeque<ProximaRecord>,
    last_oid: Option<String>,
    done: bool,
}

impl Source<'_> {
    /// The next record of this source, pulling a chunk when the buffer is empty.
    /// Fails closed when the source breaks the ordering contract: a merge over an
    /// unordered source would silently keep a stale copy of a row.
    async fn next_record(&mut self, index: usize) -> RecordStoreResult<Option<ProximaRecord>> {
        while self.buffered.is_empty() && !self.done {
            match self.stream.next().await {
                Some(chunk) => self.buffered.extend(chunk?),
                None => self.done = true,
            }
        }
        let Some(record) = self.buffered.pop_front() else {
            return Ok(None);
        };
        if let Some(last) = &self.last_oid
            && record.oid.as_str() <= last.as_str()
        {
            return Err(anyhow::anyhow!(
                "ordered record scan: source {index} yielded oid '{}' after '{last}' \
                 (sources must be strictly ascending by oid)",
                record.oid
            ));
        }
        self.last_oid = Some(record.oid.clone());
        Ok(Some(record))
    }
}

/// K-way merge of ordered chunk streams. `sources` run oldest → newest: for an
/// oid present in several, the copy from the **newest** source wins and the
/// others are dropped. The output keeps the ordering contract. Memory is one
/// chunk per source plus one output chunk.
pub fn merge_newest_wins<'a>(
    sources: Vec<RecordChunkStream<'a>>,
    chunk_rows: usize,
) -> RecordChunkStream<'a> {
    let chunk_rows = effective_chunk_rows(chunk_rows);
    if sources.len() == 1 {
        return sources.into_iter().next().expect("one source");
    }
    struct State<'a> {
        sources: Vec<Source<'a>>,
        heap: BinaryHeap<Head>,
        primed: bool,
        failed: bool,
    }
    let state = State {
        sources: sources
            .into_iter()
            .map(|stream| Source {
                stream,
                buffered: VecDeque::new(),
                last_oid: None,
                done: false,
            })
            .collect(),
        heap: BinaryHeap::new(),
        primed: false,
        failed: false,
    };
    stream::unfold(state, move |mut st| async move {
        if st.failed {
            return None;
        }
        let result: RecordStoreResult<Vec<ProximaRecord>> = async {
            if !st.primed {
                st.primed = true;
                for index in 0..st.sources.len() {
                    if let Some(record) = st.sources[index].next_record(index).await? {
                        st.heap.push(Head {
                            record,
                            source: index,
                        });
                    }
                }
            }
            let mut out = Vec::with_capacity(chunk_rows.min(1024));
            while out.len() < chunk_rows {
                let Some(winner) = st.heap.pop() else {
                    break;
                };
                // Refill the winner's source, then drop every older copy of the oid
                // (refilling each of those sources too).
                if let Some(next) = st.sources[winner.source].next_record(winner.source).await? {
                    st.heap.push(Head {
                        record: next,
                        source: winner.source,
                    });
                }
                while st
                    .heap
                    .peek()
                    .is_some_and(|h| h.record.oid == winner.record.oid)
                {
                    let shadowed = st.heap.pop().expect("peeked");
                    if let Some(next) = st.sources[shadowed.source]
                        .next_record(shadowed.source)
                        .await?
                    {
                        st.heap.push(Head {
                            record: next,
                            source: shadowed.source,
                        });
                    }
                }
                out.push(winner.record);
            }
            Ok(out)
        }
        .await;
        match result {
            Ok(chunk) if chunk.is_empty() => None,
            Ok(chunk) => Some((Ok(chunk), st)),
            Err(e) => {
                st.failed = true;
                Some((Err(e), st))
            }
        }
    })
    .boxed()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ProximaTreeNode, ProximaValue};
    use futures::TryStreamExt;

    fn rec(oid: &str, version: i64) -> ProximaRecord {
        let mut r = ProximaRecord {
            oid: oid.to_string(),
            ..Default::default()
        };
        r.props.insert(
            "v".to_string(),
            ProximaTreeNode::Value(ProximaValue::Int64(version)),
        );
        r
    }

    fn version(r: &ProximaRecord) -> i64 {
        match r.props.get("v") {
            Some(ProximaTreeNode::Value(ProximaValue::Int64(v))) => *v,
            other => panic!("unexpected {other:?}"),
        }
    }

    async fn collect(stream: RecordChunkStream<'_>) -> Vec<(String, i64)> {
        let chunks: Vec<Vec<ProximaRecord>> = stream.try_collect().await.unwrap();
        chunks
            .into_iter()
            .flatten()
            .map(|r| (r.oid.clone(), version(&r)))
            .collect()
    }

    fn source(oids: &[&str], version: i64, chunk: usize) -> RecordChunkStream<'static> {
        chunks_of_records(oids.iter().map(|o| rec(o, version)).collect(), chunk)
    }

    #[tokio::test]
    async fn the_newest_source_wins_and_order_is_kept() {
        let merged = merge_newest_wins(
            vec![
                source(&["a", "c", "e", "g"], 1, 2),
                source(&["b", "c", "g"], 2, 1),
                source(&["c", "d"], 3, 5),
            ],
            3,
        );
        assert_eq!(
            collect(merged).await,
            vec![
                ("a".into(), 1),
                ("b".into(), 2),
                ("c".into(), 3),
                ("d".into(), 3),
                ("e".into(), 1),
                ("g".into(), 2),
            ]
        );
    }

    #[tokio::test]
    async fn output_chunks_respect_the_chunk_size() {
        let merged = merge_newest_wins(
            vec![source(&["a", "b", "c", "d", "e"], 1, 5), source(&[], 2, 5)],
            2,
        );
        let sizes: Vec<usize> = merged
            .map_ok(|c| c.len())
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(sizes, vec![2, 2, 1]);
    }

    #[tokio::test]
    async fn an_unordered_source_fails_closed() {
        let unordered: RecordChunkStream<'static> =
            stream::iter(vec![Ok(vec![rec("b", 1), rec("a", 1)])]).boxed();
        let merged = merge_newest_wins(vec![unordered, source(&["c"], 2, 1)], 10);
        let result: RecordStoreResult<Vec<Vec<ProximaRecord>>> = merged.try_collect().await;
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("strictly ascending")
        );
    }

    #[tokio::test]
    async fn key_snapshot_skips_vanished_keys_and_filters() {
        let keys = vec![
            "c".to_string(),
            "a".to_string(),
            "b".to_string(),
            "a".to_string(),
        ];
        let stream = chunks_by_key_snapshot(keys, 2, |k| (k != "b").then(|| rec(k, 7)));
        assert_eq!(
            collect(stream).await,
            vec![("a".into(), 7), ("c".into(), 7)]
        );
        let filtered = filter_chunks(source(&["a", "b", "c"], 1, 1), |r| r.oid != "b");
        assert_eq!(
            collect(filtered).await,
            vec![("a".into(), 1), ("c".into(), 1)]
        );
    }

    #[tokio::test]
    async fn chunks_of_records_sorts_and_keeps_the_last_duplicate() {
        let stream = chunks_of_records(vec![rec("b", 1), rec("a", 1), rec("b", 2)], 10);
        assert_eq!(
            collect(stream).await,
            vec![("a".into(), 1), ("b".into(), 2)]
        );
    }
}
