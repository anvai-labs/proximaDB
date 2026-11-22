use super::*;
use proximadb_data_model::ProximaValue as PV;
use proximadb_records::ProximaTreeNode;
use proximadb_search_types::sql_value_filter::evaluate_filter_proxima;
use serde_json::json;

const COL: &str = "tag";
const COL2: &str = "score";

fn props(values: &[(&str, Option<PV>)]) -> ProximaTree {
    values
        .iter()
        .filter_map(|(k, v)| {
            v.clone()
                .map(|v| ((*k).to_string(), ProximaTreeNode::Value(v)))
        })
        .collect()
}

fn columns() -> Vec<PruneColumn> {
    vec![
        PruneColumn {
            name: COL.into(),
            column_id: 100,
        },
        PruneColumn {
            name: COL2.into(),
            column_id: 101,
        },
    ]
}

/// One block per entry of `blocks`; each row holds `(tag, score)`.
fn table(
    blocks: &[Vec<(Option<PV>, Option<PV>)>],
) -> (BlockPruneStatsTable, Vec<Vec<ProximaTree>>) {
    let mut builder = BlockPruneStatsBuilder::new(columns()).unwrap();
    let mut rows_by_block = Vec::new();
    for block in blocks {
        let mut rows = Vec::new();
        for (tag, score) in block {
            let p = props(&[(COL, tag.clone()), (COL2, score.clone())]);
            builder.push_row(&p);
            rows.push(p);
        }
        builder.cut_block(block.len() as u32);
        rows_by_block.push(rows);
    }
    (builder.finish().unwrap(), rows_by_block)
}

fn cmp(field: &str, op: ComparisonOperator, value: Value) -> FilterExpression {
    FilterExpression::Comparison {
        field: field.into(),
        operator: op,
        value,
    }
}

fn s(v: &str) -> Option<PV> {
    Some(PV::String(v.into()))
}
fn i(v: i64) -> Option<PV> {
    Some(PV::Int64(v))
}
fn f(v: f64) -> Option<PV> {
    Some(PV::Float64(v))
}

/// Ground truth: some row of the block satisfies the filter.
fn any_row_matches(rows: &[ProximaTree], filter: &FilterExpression) -> bool {
    rows.iter().any(|p| evaluate_filter_proxima(filter, p))
}

#[test]
fn round_trips_and_is_deterministic() {
    let (t, _) = table(&[
        vec![(s("a"), i(1)), (s("b"), i(5)), (None, f(2.5))],
        (0..40)
            .map(|k| {
                (
                    s(&format!("long-string-value-number-{k:03}")),
                    Some(PV::Null),
                )
            })
            .collect(),
    ]);
    let bytes = t.to_bytes().unwrap();
    let back = BlockPruneStatsTable::parse(&bytes).unwrap();
    assert_eq!(back, t);
    assert_eq!(back.to_bytes().unwrap(), bytes);
    // The second block's tag column overflowed the distinct cap: dropped, not
    // truncated, and a bloom took its place; its long max is not stored.
    let tag = &t.blocks[1][0];
    assert!(tag.distinct.is_none());
    assert!(tag.bloom.is_some());
    assert!(
        tag.str_max.is_none(),
        "a truncated upper bound is never stored"
    );
    assert_eq!(
        tag.str_min.as_deref().map(str::len),
        Some(STRING_BOUND_PREFIX_BYTES)
    );
}

#[test]
fn parse_fails_closed_on_every_truncation_and_on_trailing_bytes() {
    let (t, _) = table(&[vec![(s("a"), i(1)), (s("zz"), f(9.0))], vec![(None, None)]]);
    let bytes = t.to_bytes().unwrap();
    for len in 0..bytes.len() {
        assert!(
            BlockPruneStatsTable::parse(&bytes[..len]).is_err(),
            "truncation to {len} bytes must fail"
        );
    }
    let mut trailing = bytes.clone();
    trailing.push(0);
    assert!(BlockPruneStatsTable::parse(&trailing).is_err());
    let mut version = bytes.clone();
    version[0] = 2;
    assert!(BlockPruneStatsTable::parse(&version).is_err());
}

#[test]
fn single_bit_flips_never_parse_into_a_table_that_prunes_a_matching_block() {
    // A bit flip either fails the parse or yields a table; the directory and
    // footer carry checksums for the bytes as a whole, so this only pins that
    // parse never panics and that a surviving table still parses its rows.
    let (t, _) = table(&[vec![(s("a"), i(1)), (s("b"), i(3))], vec![(s("c"), i(7))]]);
    let bytes = t.to_bytes().unwrap();
    for byte in 0..bytes.len() {
        for bit in 0..8 {
            let mut flipped = bytes.clone();
            flipped[byte] ^= 1 << bit;
            let _ = BlockPruneStatsTable::parse(&flipped);
        }
    }
}

#[test]
fn equality_range_and_in_prune_only_provably_empty_blocks() {
    let (t, rows) = table(&[
        vec![(s("red"), i(1)), (s("red"), i(4))],
        vec![(s("blue"), i(10)), (s("green"), i(12))],
    ]);
    // Positive controls: these prune.
    let eq_blue = cmp(COL, ComparisonOperator::Equals, json!("blue"));
    assert!(!t.block_may_match(0, &eq_blue));
    assert!(t.block_may_match(1, &eq_blue));
    let gt = cmp(COL2, ComparisonOperator::GreaterThan, json!(4));
    assert!(!t.block_may_match(0, &gt), "max 4 is not > 4");
    assert!(t.block_may_match(1, &gt));
    let between = cmp(COL2, ComparisonOperator::Between, json!([5, 9]));
    assert!(!t.block_may_match(0, &between));
    assert!(!t.block_may_match(1, &between));
    let in_list = cmp(COL, ComparisonOperator::In, json!(["green", "black"]));
    assert!(!t.block_may_match(0, &in_list));
    assert!(t.block_may_match(1, &in_list));
    // Negative controls: a block holding a match is kept, whatever the stat.
    for filter in [&eq_blue, &gt, &between, &in_list] {
        for (b, block) in rows.iter().enumerate() {
            if any_row_matches(block, filter) {
                assert!(t.block_may_match(b, filter), "block {b} holds a match");
            }
        }
    }
    // A value inside the bounds but not in the distinct set prunes (the
    // distinct set is exact); a float literal equal to an integer keeps.
    let eq_two = cmp(COL2, ComparisonOperator::Equals, json!(2));
    assert!(!t.block_may_match(0, &eq_two));
    let eq_four_float = cmp(COL2, ComparisonOperator::Equals, json!(4.0));
    assert!(
        t.block_may_match(0, &eq_four_float),
        "4 == 4.0 in the evaluator"
    );
    // MUTATION PROBE (pinned): the evaluator's relative-epsilon equality spans
    // 4.0's 1-ULP neighbours (the mut_probe test pins 4.0 == 3.9999999999999996),
    // so a block holding 4.0 must be kept for those literals — only
    // slack-aware bounds may prune them.
    // 11 distinct values (so the distinct set is DROPPED and only the bounds
    // speak), one of them 4.0.
    let block0: Vec<_> = std::iter::once(f(4.0))
        .chain((5..=15).map(|k| f(k as f64)))
        .map(|v| (None, v))
        .collect();
    let (f_t, f_rows) = table(&[block0]);
    assert!(f_t.blocks[0][1].distinct.is_none(), "bounds-only block");
    let eq = cmp(COL2, ComparisonOperator::Equals, json!(3.9999999999999996));
    assert!(
        f_t.block_may_match(0, &eq),
        "evaluator matches 3.9999999999999996 against the 4.0 row; bounds-only pruning must keep"
    );
    assert!(any_row_matches(&f_rows[0], &eq), "positive control");
}

#[test]
fn null_and_absent_rows_are_decided_by_the_evaluator() {
    let (t, rows) = table(&[
        vec![(s("x"), i(1)), (s("y"), i(2))],
        vec![(None, i(1)), (s("x"), Some(PV::Null))],
    ]);
    let is_null = cmp(COL, ComparisonOperator::IsNull, Value::Null);
    assert!(
        !t.block_may_match(0, &is_null),
        "no null or absent tag in block 0"
    );
    assert!(t.block_may_match(1, &is_null), "block 1 has an absent tag");
    let score_null = cmp(COL2, ComparisonOperator::Equals, Value::Null);
    assert!(!t.block_may_match(0, &score_null));
    assert!(
        t.block_may_match(1, &score_null),
        "present-null score in block 1"
    );
    let not_exists = cmp(COL, ComparisonOperator::NotExists, Value::Null);
    assert!(t.block_may_match(1, &not_exists));
    let exists = cmp(COL2, ComparisonOperator::IsNotNull, Value::Null);
    assert!(t.block_may_match(0, &exists));
    for filter in [&is_null, &score_null, &not_exists, &exists] {
        for (b, block) in rows.iter().enumerate() {
            assert_eq!(
                t.block_may_match(b, filter),
                any_row_matches(block, filter),
                "null tests are exact here (distinct sets are complete): block {b} {filter:?}"
            );
        }
    }
}

#[test]
fn a_long_string_upper_bound_never_prunes() {
    // > 8 distinct strings so only the bounds (and bloom) speak.
    let block: Vec<_> = (0..12)
        .map(|k| (s(&format!("aaaaaaaaaaaaaaaaaaaa{k:02}")), None))
        .chain([(s("zzzzzzzzzzzzzzzzzzzzzz-long"), None)])
        .collect();
    let (t, rows) = table(&[block]);
    let gt = cmp(
        COL,
        ComparisonOperator::GreaterThan,
        json!("zzzzzzzzzzzzzzzz"),
    );
    assert!(any_row_matches(&rows[0], &gt));
    assert!(
        t.block_may_match(0, &gt),
        "truncated max must not prune `>`"
    );
    let lt = cmp(COL, ComparisonOperator::LessThan, json!("a"));
    assert!(
        !t.block_may_match(0, &lt),
        "the truncated min is a valid lower bound"
    );
}

#[test]
fn distinct_overflow_and_bloom_false_positives_keep() {
    // 20 distinct strings: the distinct set is dropped, a bloom is built.
    let names: Vec<String> = (0..20).map(|k| format!("user-{k}")).collect();
    let block: Vec<_> = names.iter().map(|n| (s(n), None)).collect();
    let (t, rows) = table(&[block]);
    assert!(t.blocks[0][0].distinct.is_none());
    let bloom = t.blocks[0][0].bloom.clone().unwrap();
    for n in &names {
        let eq = cmp(COL, ComparisonOperator::Equals, json!(n));
        assert!(t.block_may_match(0, &eq), "member {n} must be kept");
    }
    // Find a bloom false positive among non-members inside the string bounds:
    // it must be kept (the bloom answered "maybe"), and most non-members prune.
    let mut false_positive = None;
    let mut pruned = 0;
    for k in 0..5000 {
        // Inside the string bounds ["user-0", "user-9"], so only the bloom
        // can speak.
        let probe = format!("user-5{k}x");
        let eq = cmp(COL, ComparisonOperator::Equals, json!(probe));
        assert!(!any_row_matches(&rows[0], &eq));
        if proximadb_block_format::bloom_may_contain_hash(
            &bloom,
            proximadb_block_format::header::fnv1a_hash(&probe),
        ) {
            assert!(
                t.block_may_match(0, &eq),
                "bloom positive {probe} must be kept"
            );
            false_positive.get_or_insert(probe);
        } else {
            assert!(!t.block_may_match(0, &eq));
            pruned += 1;
        }
    }
    assert!(
        false_positive.is_some(),
        "the probe set must hit a false positive"
    );
    assert!(
        pruned > 4000,
        "the bloom must prune most non-members ({pruned})"
    );
}

#[test]
fn cross_class_predicates_keep() {
    // Under the evaluator's total order a string outranks any number, so
    // `tag > 5` matches every string — never inferred from string bounds.
    let (t, rows) = table(&[vec![(s("abc"), f(1.5)), (s("abd"), f(2.5))]]);
    let gt_num = cmp(COL, ComparisonOperator::GreaterThan, json!(5));
    assert!(any_row_matches(&rows[0], &gt_num));
    assert!(t.block_may_match(0, &gt_num));
    let lt_str = cmp(COL2, ComparisonOperator::LessThan, json!("a"));
    assert!(any_row_matches(&rows[0], &lt_str));
    assert!(t.block_may_match(0, &lt_str));
}

#[test]
fn builder_poisons_on_a_block_cut_that_disagrees_with_the_rows() {
    let mut b = BlockPruneStatsBuilder::new(columns()).unwrap();
    b.push_row(&props(&[(COL, s("a"))]));
    b.cut_block(2);
    assert!(b.finish().is_none());
    let mut b = BlockPruneStatsBuilder::new(columns()).unwrap();
    b.push_row(&props(&[(COL, s("a"))]));
    assert!(b.finish().is_none(), "rows after the last cut emit nothing");
    assert!(BlockPruneStatsBuilder::new(Vec::new()).is_none());
}

#[test]
fn check_block_rows_pins_the_block_table() {
    let (t, _) = table(&[vec![(s("a"), None)], vec![(s("b"), None), (s("c"), None)]]);
    assert!(t.check_block_rows(&[1, 2]).is_ok());
    assert!(t.check_block_rows(&[2, 1]).is_err());
    assert!(t.check_block_rows(&[1]).is_err());
}

// ---------------------------------------------------------------------------
// Randomized soundness property: for every generated block and filter, a block
// holding a row the evaluator matches is never pruned.
// ---------------------------------------------------------------------------

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

fn value_pool(kind: usize, rng: &mut Rng) -> Option<PV> {
    let strings = [
        "a",
        "b",
        "ab",
        "red",
        "blue",
        "zz",
        "",
        "user-1",
        "user-22",
        "aaaaaaaaaaaaaaaaaaaaaaaa",
        "aaaaaaaaaaaaaaaaz",
        "zzzzzzzzzzzzzzzzzzzz",
    ];
    let pick_int = |rng: &mut Rng| [-5i64, -1, 0, 1, 2, 3, 4, 7, 100, (1 << 60) + 1][rng.below(10)];
    let pick_float =
        // 4.0 and its 1-ULP neighbours: the evaluator's relative-epsilon
        // equality spans them, so only slack-aware bounds may prune them.
        |rng: &mut Rng| {
            [-2.5f64, 0.0, 0.5, 1.0, 1.5, 3.0, 4.0, 4.000000000000001, 3.9999999999999996, 1e20]
                [rng.below(10)]
        };
    match kind {
        0 => None,
        1 => Some(PV::Null),
        2 => Some(PV::Int64(pick_int(rng))),
        3 => Some(PV::Float64(pick_float(rng))),
        4 => Some(PV::String(strings[rng.below(strings.len())].to_string())),
        5 => Some(PV::Boolean(rng.below(2) == 0)),
        6 => Some(PV::UInt64(u64::MAX - rng.below(3) as u64)),
        7 => Some(PV::String(format!("id-{:04}", rng.below(300)))),
        _ => Some(PV::Array(vec![
            PV::Int64(pick_int(rng)),
            PV::String("a".into()),
        ])),
    }
}

fn literal_pool(rng: &mut Rng) -> Value {
    let pool = [
        json!(null),
        json!(0),
        json!(1),
        json!(3),
        json!(4),
        json!(4.0),
        json!(4.000000000000001),
        json!(3.9999999999999996),
        json!(-5),
        json!(1.5),
        json!(100),
        json!(1152921504606846977i64),
        json!(1152921504606846976.0),
        json!(u64::MAX),
        json!(true),
        json!("a"),
        json!("ab"),
        json!("red"),
        json!("zz"),
        json!(""),
        json!("aaaaaaaaaaaaaaaaz"),
        json!("aaaaaaaaaaaaaaaaaaaaaaaa"),
        json!("zzzzzzzzzzzzzzzzzzzzz"),
        json!("id-0042"),
        json!("id-0299"),
        json!("a%"),
        json!("^a"),
    ];
    pool[rng.below(pool.len())].clone()
}

fn random_leaf(rng: &mut Rng) -> FilterExpression {
    use ComparisonOperator as Op;
    let ops = [
        Op::Equals,
        Op::NotEquals,
        Op::GreaterThan,
        Op::GreaterThanOrEqual,
        Op::LessThan,
        Op::LessThanOrEqual,
        Op::In,
        Op::NotIn,
        Op::Contains,
        Op::StartsWith,
        Op::EndsWith,
        Op::Between,
        Op::IsNull,
        Op::IsNotNull,
        Op::Like,
        Op::Exists,
        Op::NotExists,
        Op::Regex,
    ];
    let op = ops[rng.below(ops.len())].clone();
    let value = match op {
        Op::In | Op::NotIn => {
            let n = 1 + rng.below(3);
            Value::Array((0..n).map(|_| literal_pool(rng)).collect())
        }
        Op::Between => json!([literal_pool(rng), literal_pool(rng)]),
        _ => literal_pool(rng),
    };
    let field = if rng.below(2) == 0 { COL } else { COL2 };
    cmp(field, op, value)
}

fn random_filter(rng: &mut Rng, depth: usize) -> FilterExpression {
    if depth == 0 || rng.below(3) == 0 {
        return random_leaf(rng);
    }
    match rng.below(3) {
        0 => FilterExpression::And(vec![
            random_filter(rng, depth - 1),
            random_filter(rng, depth - 1),
        ]),
        1 => FilterExpression::Or(vec![
            random_filter(rng, depth - 1),
            random_filter(rng, depth - 1),
        ]),
        _ => FilterExpression::Not(Box::new(random_filter(rng, depth - 1))),
    }
}

#[test]
fn mut_probe_evaluator_float_equality() {
    use proximadb_search_types::sql_value_filter::evaluate_filter_resolved;
    let f = FilterExpression::Comparison {
        field: "x".into(),
        operator: ComparisonOperator::Equals,
        value: serde_json::json!(3.9999999999999996),
    };
    for row in [
        json!(4.0),
        json!(4),
        json!(4.000000000000001),
        json!(3.9999999999999996),
    ] {
        let m = evaluate_filter_resolved(&f, &|_: &str| Some(row.clone()));
        eprintln!("PROBE {row} matches: {m}");
    }
}

#[test]
fn randomized_pruning_never_drops_a_block_holding_a_match() {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let mut pruned = 0usize;
    let mut checked = 0usize;
    for _ in 0..300 {
        // Each block draws its rows from one or two value kinds, so both
        // homogeneous (bounds, bloom) and mixed (keep) blocks occur.
        let n_blocks = 1 + rng.below(3);
        let blocks: Vec<Vec<(Option<PV>, Option<PV>)>> = (0..n_blocks)
            .map(|_| {
                let rows = 1 + rng.below(40);
                let kinds = [rng.below(9), rng.below(9)];
                let mixed = rng.below(4) == 0;
                (0..rows)
                    .map(|_| {
                        let k0 = if mixed { kinds[rng.below(2)] } else { kinds[0] };
                        let k1 = if mixed { kinds[rng.below(2)] } else { kinds[1] };
                        (value_pool(k0, &mut rng), value_pool(k1, &mut rng))
                    })
                    .collect()
            })
            .collect();
        let (t, rows) = table(&blocks);
        let t = BlockPruneStatsTable::parse(&t.to_bytes().unwrap()).unwrap();
        for _ in 0..60 {
            let filter = random_filter(&mut rng, 2);
            for (b, block) in rows.iter().enumerate() {
                checked += 1;
                let may = t.block_may_match(b, &filter);
                if any_row_matches(block, &filter) {
                    assert!(
                        may,
                        "UNSOUND prune: block {b} holds a match for {filter:?}\nrows: {block:?}\nstats: {:?}",
                        t.blocks[b]
                    );
                }
                if !may {
                    pruned += 1;
                }
            }
        }
    }
    // Non-vacuity: the property must exercise real prunes.
    assert!(
        pruned * 10 > checked,
        "too few prunes to be meaningful: {pruned}/{checked}"
    );
}
