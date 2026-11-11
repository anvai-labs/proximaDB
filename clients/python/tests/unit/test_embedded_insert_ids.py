"""Regression tests for the embedded id echo (PR #2500 follow-up).

The Rust PyO3 binding's write contract (proximadb-embedded python.rs):
insert/insert_numpy return a processed count (int); upsert/upsert_numpy
return an (inserted, updated) TUPLE. These tests stub that real shape —
earlier test doubles stubbed ints for upserts, masking a dead echo.
"""

from proximadb_sdk.adapters.embedded_adapter import EmbeddedProtocolAdapter


def _adapter(insert_result=None, upsert_result=None):
    # skip __init__ (it loads the PyO3 binding, absent in the test env);
    # stub _db with the REAL binding contract shapes: writes return a
    # processed count (insert) or an (inserted, updated) tuple (upsert).
    adapter = EmbeddedProtocolAdapter.__new__(EmbeddedProtocolAdapter)
    adapter._db = type(
        "Db",
        (),
        {
            "insert_numpy": lambda self, *a, **k: (
                insert_result if insert_result is not None else len(a[1])
            ),
            "upsert_numpy": lambda self, *a, **k: (
                upsert_result if upsert_result is not None else (len(a[1]), 0)
            ),
        },
    )()
    adapter._connected = True
    return adapter


def _records(n, with_ids=True):
    return [
        {
            **({"id": f"r{i}"} if with_ids else {}),
            "vector": [float(i), 1.0, 0.0, 0.0],
            "metadata": {},
        }
        for i in range(n)
    ]


def test_insert_numpy_echoes_ids_on_full_success():
    adapter = _adapter(insert_result=3)
    ids = ["a", "b", "c"]
    resp = adapter.insert_numpy("col", ids, [[1.0] * 4] * 3, None)
    assert resp.vector_ids == ids


def test_upsert_numpy_echoes_ids_on_full_success():
    # REAL binding contract: upsert_numpy returns an (inserted, updated)
    # TUPLE — an int stub masks the echo path (the round-8 review's finding).
    # (0, 2) is the discriminating all-UPDATED case: the pre-PR counting
    # (inserted component only) would report total failure and suppress the
    # echo — this test goes red under that regression.
    adapter = _adapter(upsert_result=(0, 2))
    records = _records(2)
    resp = adapter.upsert_numpy(
        "col", [r["id"] for r in records], [r["vector"] for r in records], None
    )
    assert resp.vector_ids == [r["id"] for r in records]


def test_partial_success_suppresses_the_echo():
    # REACHABLE partial shape: 3 ids paired with 3 vectors, the binding
    # reports 2 processed (one record rejected server-side) → failed=1 →
    # the echo MUST be suppressed
    adapter = _adapter(insert_result=2)
    ids = ["a", "b", "c"]
    resp = adapter.insert_numpy("col", ids, [[1.0] * 4] * 3, None)
    assert resp.vector_ids == []


def test_successful_count_helper_is_defensive_on_malformed_tuples():
    # a binding upgrade adding a non-int category must not convert a
    # committed write into a reported failure: non-int parts are ignored,
    # and a tuple with no int parts counts as full success (the total
    # fallback) rather than crashing the echo gate.
    from proximadb_sdk.adapters.embedded_adapter import _successful_write_count

    # (2, "err") sums its int parts → 2: the helper trusts int parts and
    # ignores non-int ones (the documented defensive behavior — a malformed
    # shape must not crash a committed write, and must not fabricate extra
    # successes either)
    assert _successful_write_count((2, "err"), 3) == 2
    assert _successful_write_count(3, 3) == 3


def test_empty_batch_is_a_clean_noop():
    adapter = _adapter(insert_result=0)
    resp = adapter.insert_numpy("col", [], [], None)
    assert resp.success is True
    assert resp.vector_ids == []
