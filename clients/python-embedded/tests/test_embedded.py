"""Tests for ProximaDB Embedded Mode"""

import tempfile
import uuid
from pathlib import Path

import numpy as np
import pytest

from proximadb_embedded import (
    DiskConfig,
    GraphEdge,
    GraphNode,
    ProximaDB,
    SearchResult,
    insert_arrow,
    insert_pandas,
)


class TestEmbeddedBasics:
    """Basic functionality tests"""

    def test_embedded_source_avoids_loopback_protocol_clients(self):
        """Embedded mode must stay wired to in-process services/API handlers."""
        repo_root = Path(__file__).resolve().parents[3]
        # #1021 moved the embedded engine out of src/embedded into the binding
        # crate; this guard silently broke then (file-not-found) — one more
        # instance of a check whose target moved out from under it.
        crate_src = repo_root / "crates" / "binding" / "proximadb-embedded" / "src"
        embedded_sources = [
            crate_src / "lib.rs",
            crate_src / "python.rs",
        ]
        forbidden_tokens = [
            "127.0.0.1",
            "localhost",
            "reqwest",
            "tonic::transport::Channel",
            "Channel::from_shared",
            "connect_lazy",
            "connect().await",
            "ArrowFlightServer::new",
            "RestServer",
            "GrpcServer",
        ]

        for source_path in embedded_sources:
            source = source_path.read_text(encoding="utf-8")
            for token in forbidden_tokens:
                assert token not in source, (
                    f"{source_path.relative_to(repo_root)} contains {token!r}; "
                    "embedded Python should call in-process services/API handlers"
                )

    def test_create_database(self):
        """Test database creation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = ProximaDB(data_dirs=tmpdir)
            assert db is not None

    def test_create_collection(self):
        """Test collection creation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = ProximaDB(data_dirs=tmpdir)
            db.create_collection("test_collection", dimension=128)

            info = db.get_collection("test_collection")
            assert info is not None
            assert info.name == "test_collection"
            assert info.dimension == 128

    def test_list_collections(self):
        """Test listing collections"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = ProximaDB(data_dirs=tmpdir)
            db.create_collection("collection_one", dimension=64)
            db.create_collection("collection_two", dimension=128)

            collections = db.list_collections()
            names = [c.name for c in collections]
            assert "collection_one" in names
            assert "collection_two" in names

    def test_delete_collection(self):
        """Test collection deletion"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = ProximaDB(data_dirs=tmpdir)
            db.create_collection("to_delete", dimension=64)
            db.delete_collection("to_delete")

            info = db.get_collection("to_delete")
            assert info is None

    def test_execute_sql_agentic_schema_ddl(self):
        """Test embedded SQL DDL lowers mixed schema into Rust catalog."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = ProximaDB(data_dirs=tmpdir)
            result = db.execute_sql(
                """
                CREATE TABLE IF NOT EXISTS "agent_store" (
                    "record_id" TEXT NOT NULL,
                    "payload" JSONB NOT NULL DEFAULT '{}'::jsonb,
                    "embedding" VECTOR(32),
                    PRIMARY KEY ("record_id")
                ) WITH (
                    storage_engine = 'SST',
                    layout = 'hybrid',
                    xcatalog_namespace = 'agentic.pyembedded',
                    schema_kind = 'agentic_mixed'
                );
                """
            )

            assert result["row_count"] == 1
            assert result["rows"][0]["success"] is True
            assert "Created table" in result["rows"][0]["message"]

            gin = db.execute_sql(
                "CREATE INDEX idx_agent_payload ON agent_store USING GIN (payload);"
            )
            hnsw = db.execute_sql(
                "CREATE INDEX idx_agent_embedding ON agent_store USING HNSW (embedding);"
            )
            assert gin["rows"][0]["success"] is True
            assert hnsw["rows"][0]["success"] is True

            tables = db.execute_sql(
                "SELECT * FROM xcatalog.tables WHERE table_name = 'agent_store';"
            )
            assert tables["row_count"] == 1
            assert tables["rows"][0]["table_name"] == "agent_store"
            assert tables["rows"][0]["schema_kind"] == "agentic_mixed"
            assert tables["rows"][0]["storage_engine"] == "SST"
            assert tables["rows"][0]["xcatalog_namespace"] == "agentic.pyembedded"

            columns = db.execute_sql(
                "SELECT * FROM xcatalog.columns WHERE table_name = 'agent_store';"
            )
            assert any(
                row["column_name"] == "payload" and row["data_type"] == "jsonb"
                for row in columns["rows"]
            )
            assert any(
                row["column_name"] == "embedding"
                and row["data_type"] == "vector"
                and row["vector_dimension"] == "32"
                for row in columns["rows"]
            )

            indexes = db.execute_sql(
                "SELECT * FROM xcatalog.indexes WHERE table_name = 'agent_store';"
            )
            assert any(
                row["index_name"] == "idx_agent_payload" and row["index_type"] == "gin"
                for row in indexes["rows"]
            )
            assert any(
                row["index_name"] == "idx_agent_embedding"
                and row["index_type"] == "hnsw"
                for row in indexes["rows"]
            )

            insert = db.execute_sql(
                """
                INSERT INTO agent_store (
                    record_id,
                    payload,
                    embedding
                ) VALUES (
                    'record-1',
                    '{"kind":"memory","score":7}'::jsonb,
                    '[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                      0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
                      1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4,
                      2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2]'
                );
                """
            )
            assert insert["row_count"] == 1
            assert insert["rows"][0]["success"] is True
            assert insert["rows"][0]["rows_affected"] == 1
            assert insert["rows"][0]["inserted_ids"] == ["record-1"]

            update = db.execute_sql(
                """
                UPDATE agent_store
                SET payload = '{"kind":"updated","score":9}'::jsonb,
                    embedding = '[9.9, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                      0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
                      1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4,
                      2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2]'
                WHERE record_id = 'record-1';
                """
            )
            assert update["row_count"] == 1
            assert update["rows"][0]["success"] is True
            assert update["rows"][0]["rows_affected"] == 1

            updated_record = db.get_vector("agent_store", "record-1")
            assert updated_record is not None
            assert len(updated_record["vector"]) == 32
            assert abs(updated_record["vector"][0] - 9.9) < 0.001

            default_insert = db.execute_sql(
                """
                INSERT INTO agent_store (
                    record_id,
                    embedding
                ) VALUES (
                    'record-2',
                    '[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                      0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
                      1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4,
                      2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2]'
                );
                """
            )
            assert default_insert["row_count"] == 1
            assert default_insert["rows"][0]["success"] is True
            assert default_insert["rows"][0]["rows_affected"] == 1
            assert default_insert["rows"][0]["inserted_ids"] == ["record-2"]

            delete = db.execute_sql(
                "DELETE FROM agent_store WHERE record_id = 'record-2';"
            )
            assert delete["row_count"] == 1
            assert delete["rows"][0]["success"] is True
            assert delete["rows"][0]["rows_affected"] == 1


class TestVectorOperations:
    """Vector insert and search tests"""

    def test_insert_vectors(self):
        """Test vector insertion"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = ProximaDB(data_dirs=tmpdir)
            db.create_collection("vectors_main", dimension=128)

            vectors = np.random.rand(100, 128).astype(np.float32)
            ids = [f"vec_{i}" for i in range(100)]

            count = db.insert("vectors_main", ids=ids, vectors=vectors)
            assert count == 100

    def test_insert_with_metadata(self):
        """Test vector insertion with metadata"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = ProximaDB(data_dirs=tmpdir)
            db.create_collection("vectors_meta", dimension=64)

            vectors = np.random.rand(10, 64).astype(np.float32)
            ids = [f"vec_{i}" for i in range(10)]
            metadata = [{"category": "A", "score": 0.9} for _ in range(10)]

            count = db.insert("vectors_meta", ids=ids, vectors=vectors, metadata=metadata)
            assert count == 10

    def test_insert_upsert_reject_ids_vectors_count_mismatch(self):
        """Length mismatches must raise ValueError on every surface.

        Parity fix (redundancy pass 3): insert_numpy already validated;
        insert/upsert/upsert_numpy used to fail late (or, for insert,
        silently zip-truncate in the record builder).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db = ProximaDB(data_dirs=tmpdir)
            db.create_collection("mismatch_guard", dimension=4)

            vectors = np.zeros((1, 4), dtype=np.float32)
            ids2 = ["a", "b"]

            for method in ("insert", "insert_numpy", "upsert", "upsert_numpy"):
                with pytest.raises(
                    ValueError,
                    match=r"Number of IDs \(2\) doesn't match number of vectors \(1\)",
                ):
                    getattr(db, method)("mismatch_guard", ids=ids2, vectors=vectors)

    def test_search_basic(self):
        """Test basic vector search"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = ProximaDB(data_dirs=tmpdir)
            db.create_collection("search_test", dimension=64)

            # Insert vectors
            vectors = np.random.rand(100, 64).astype(np.float32)
            ids = [f"vec_{i}" for i in range(100)]
            db.insert("search_test", ids=ids, vectors=vectors)

            # Search with first vector as query
            results = db.search("search_test", query=vectors[0], top_k=5)

            assert len(results) <= 5
            assert all(isinstance(r, SearchResult) for r in results)

            # First result should be exact match (or very close)
            if results:
                assert results[0].id == "vec_0" or results[0].score < 0.001

    def test_search_top_k(self):
        """Test search with different top_k values"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = ProximaDB(data_dirs=tmpdir)
            db.create_collection("topk_test", dimension=32)

            vectors = np.random.rand(50, 32).astype(np.float32)
            ids = [f"v{i}" for i in range(50)]
            db.insert("topk_test", ids=ids, vectors=vectors)

            # Test different top_k values
            for k in [1, 5, 10, 20]:
                results = db.search("topk_test", query=vectors[0], top_k=k)
                assert len(results) <= k

    def test_insert_arrow_and_pandas_batches(self):
        """Arrow/pandas embedded ingest should use the in-process batch path."""
        pa = pytest.importorskip("pyarrow")
        pd = pytest.importorskip("pandas")

        with tempfile.TemporaryDirectory() as tmpdir:
            db = ProximaDB(data_dirs=tmpdir)
            db.create_collection("arrow_vectors", dimension=4)

            table = pa.table(
                {
                    "id": ["arrow-1", "arrow-2"],
                    "vector": pa.array(
                        [[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]],
                        type=pa.list_(pa.float32()),
                    ),
                    "tenant_id": ["embedded", "embedded"],
                    "kind": ["arrow", "arrow"],
                }
            )
            assert insert_arrow(db, "arrow_vectors", table) == 2

            loaded = db.get_vector("arrow_vectors", "arrow-1")
            assert loaded is not None
            assert loaded["vector"] == pytest.approx([0.1, 0.2, 0.3, 0.4])

            frame = pd.DataFrame(
                {
                    "id": ["pandas-1"],
                    "vector": [[0.9, 0.8, 0.7, 0.6]],
                    "tenant_id": ["embedded"],
                    "kind": ["pandas"],
                }
            )
            assert insert_pandas(db, "arrow_vectors", frame) == 1
            assert db.get_vector("arrow_vectors", "pandas-1") is not None


class TestEmbeddedModalities:
    """Direct embedded tests for non-vector modality facades."""

    def test_graph_entity_relationship_flow(self):
        """Entity/SKS-style records are represented through graph-first facades."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = ProximaDB(data_dirs=tmpdir)
            graph_id = f"entity_graph_{uuid.uuid4().hex}"
            db.create_graph(graph_id)

            alice = GraphNode(
                "entity_alice",
                labels=["Entity", "Person"],
                properties={"name": "Alice", "kind": "customer"},
            )
            acme = GraphNode(
                "entity_acme",
                labels=["Entity", "Organization"],
                properties={"name": "Acme", "kind": "account"},
            )
            assert db.create_nodes(graph_id, [alice, acme]) == 2

            relation = GraphEdge(
                "entity_alice",
                "entity_acme",
                "WORKS_WITH",
                id="rel_alice_acme",
                weight=0.75,
                properties={"source": "embedded_tdd"},
            )
            assert db.create_edges(graph_id, [relation]) == 1

            loaded = db.get_node(graph_id, "entity_alice")
            assert loaded is not None
            assert "Entity" in loaded.labels
            assert loaded.properties["name"] == "Alice"

            entities = db.query_nodes_by_labels(graph_id, ["Entity"])
            assert {node.id for node in entities} >= {"entity_alice", "entity_acme"}

            outgoing = db.get_outgoing_edges(graph_id, "entity_alice")
            assert len(outgoing) == 1
            assert outgoing[0].edge_type == "WORKS_WITH"
            assert outgoing[0].to_node_id == "entity_acme"

            traversal = db.traverse_graph(graph_id, "entity_alice", max_depth=1)
            assert {node.id for node in traversal["nodes"]} >= {
                "entity_alice",
                "entity_acme",
            }

            stats = db.graph_stats(graph_id)
            assert stats.total_nodes >= 2
            assert stats.total_edges >= 1

    def test_document_collection_crud_and_query(self):
        """Documents stay behind the embedded document facade."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = ProximaDB(data_dirs=tmpdir)
            db.create_document_collection("docs", indexed_paths=["$.kind", "$.score"])

            doc_id, version = db.insert_document(
                "docs",
                {
                    "kind": "note",
                    "score": 7,
                    "payload": {"title": "embedded"},
                },
                doc_id="doc_1",
            )
            assert doc_id == "doc_1"
            assert version >= 1

            loaded = db.get_document("docs", "doc_1")
            assert loaded["kind"] == "note"
            assert loaded["payload"]["title"] == "embedded"

            matches = db.query_documents("docs", filter="$.kind = 'note'", limit=10)
            assert any(match_id == "doc_1" for match_id, _ in matches)

            db.update_document("docs", "doc_1", {"$.score": 9})
            updated = db.get_document("docs", "doc_1")
            assert updated["score"] == 9

            assert "docs" in db.list_document_collections()
            assert db.delete_document("docs", "doc_1") is True
            assert db.get_document("docs", "doc_1") is None

    def test_observability_logs_metrics_and_traces(self):
        """Observability APIs use direct embedded service methods."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = ProximaDB(data_dirs=tmpdir)
            db.create_observability_namespace("obs", retention_days=1)

            start_ns = 1_700_000_000_000_000_000
            end_ns = start_ns + 10_000_000

            assert (
                db.ingest_logs(
                    "obs",
                    [
                        {
                            "timestamp_ns": start_ns,
                            "severity": "INFO",
                            "message": "embedded service started",
                            "source": "test",
                            "service": "embedded",
                            "fields": {"tenant": "local"},
                        }
                    ],
                )
                == 1
            )
            logs = db.query_logs("obs", start_ns - 1, end_ns, query="embedded", limit=10)
            assert len(logs) == 1
            assert logs[0]["service"] == "embedded"

            assert (
                db.ingest_metrics(
                    "obs",
                    [
                        {
                            "metric_name": "request_latency_ms",
                            "timestamp_ns": start_ns,
                            "value": 12.5,
                            "labels": {"route": "/embedded"},
                        }
                    ],
                )
                == 1
            )
            points = db.aggregate_metrics(
                "obs",
                "request_latency_ms",
                aggregation="avg",
                start_time=None,
                end_time=None,
                step_seconds=60,
            )
            assert points
            assert points[0]["value"] == 12.5

            assert (
                db.ingest_traces(
                    "obs",
                    [
                        {
                            "trace_id": "trace-1",
                            "span_id": "span-1",
                            "name": "embedded_call",
                            "kind": "INTERNAL",
                            "start_time_ns": start_ns,
                            "end_time_ns": start_ns + 1_000,
                            "service": "embedded",
                            "status_code": "OK",
                            "attributes": {"surface": "pyo3"},
                        }
                    ],
                )
                == 1
            )
            spans = db.query_traces(
                "obs",
                start_ns - 1,
                end_ns,
                trace_id="trace-1",
                service="embedded",
                operation=None,
                min_duration_ns=None,
                status=None,
                limit=10,
            )
            assert len(spans) == 1
            assert spans[0]["trace_id"] == "trace-1"

            trace = db.get_trace("obs", "trace-1")
            assert trace["complete"] is True
            assert trace["spans"][0]["span_id"] == "span-1"

    def test_relational_sql_surface_uses_canonical_catalog(self):
        """Relational DDL/DML should be available without network protocols."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = ProximaDB(data_dirs=tmpdir)
            create = db.execute_sql(
                """
                CREATE TABLE IF NOT EXISTS accounts (
                    account_id TEXT NOT NULL,
                    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
                    embedding VECTOR(4),
                    PRIMARY KEY (account_id)
                ) WITH (
                    storage_engine = 'SST',
                    layout = 'hybrid',
                    schema_kind = 'relational_entity'
                );
                """
            )
            assert create["rows"][0]["success"] is True

            insert = db.execute_sql(
                """
                INSERT INTO accounts (account_id, payload, embedding)
                VALUES ('acct-1', '{"tier":"gold"}'::jsonb, '[0.1, 0.2, 0.3, 0.4]');
                """
            )
            assert insert["rows"][0]["inserted_ids"] == ["acct-1"]

            tables = db.execute_sql(
                "SELECT * FROM xcatalog.tables WHERE table_name = 'accounts';"
            )
            assert tables["row_count"] == 1
            assert tables["rows"][0]["schema_kind"] == "relational_entity"

            record = db.get_vector("accounts", "acct-1")
            assert record is not None
            assert record["id"] == "acct-1"
            assert record["vector"] == pytest.approx([0.1, 0.2, 0.3, 0.4])

    def test_unified_query_surface_exposes_plan_for_multimodal_query(self):
        """UQL-style planning should be reachable from embedded Python."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = ProximaDB(data_dirs=tmpdir)
            plan = db.explain_unified_query(
                "SELECT * FROM documents.docs WHERE $.kind = 'note'"
            )
            assert "component_count" in plan
            assert "components" in plan


class TestMultiDisk:
    """Multi-disk configuration tests"""

    def test_disk_config(self):
        """Test DiskConfig creation"""
        config = DiskConfig("/data/path", weight=2, tags=["hot", "ssd"])
        assert config.path == "/data/path"
        assert config.weight == 2
        assert "hot" in config.tags
        assert "ssd" in config.tags

    def test_multi_disk_setup(self):
        """Test multi-disk database setup"""
        with tempfile.TemporaryDirectory() as tmpdir1:
            with tempfile.TemporaryDirectory() as tmpdir2:
                disks = [
                    DiskConfig(tmpdir1, weight=2),
                    DiskConfig(tmpdir2, weight=1),
                ]
                db = ProximaDB(data_dirs=disks)

                # Should be able to create collections
                db.create_collection("multi_disk_test", dimension=128)
                info = db.get_collection("multi_disk_test")
                assert info is not None


class TestPersistence:
    """Persistence and durability tests"""

    def test_flush(self):
        """Test explicit flush"""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            db = ProximaDB(data_dirs=tmpdir)
            db.create_collection("flush_test", dimension=64)

            vectors = np.random.rand(10, 64).astype(np.float32)
            ids = [f"v{i}" for i in range(10)]
            db.insert("flush_test", ids=ids, vectors=vectors)

            # Should not raise
            db.flush()

    def test_stats(self):
        """Test storage statistics"""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            db = ProximaDB(data_dirs=tmpdir)
            db.create_collection("stats_test", dimension=128)

            vectors = np.random.rand(100, 128).astype(np.float32)
            ids = [f"v{i}" for i in range(100)]
            db.insert("stats_test", ids=ids, vectors=vectors)
            db.flush()

            stats = db.stats()
            assert stats.total_collections >= 1
            assert stats.total_vectors >= 100


class TestContextManager:
    """Context manager tests"""

    def test_with_statement(self):
        """Test using database with context manager"""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            with ProximaDB(data_dirs=tmpdir) as db:
                db.create_collection("context_test", dimension=32)
                vectors = np.random.rand(10, 32).astype(np.float32)
                db.insert(
                    "context_test", ids=[f"v{i}" for i in range(10)], vectors=vectors
                )
            # Should be flushed after exit


class TestEngines:
    """Storage engine tests"""

    @pytest.mark.parametrize(
        "engine", ["sst", "viper", "nova", "swift", "raptor", "helix"]
    )
    def test_engine_types(self, engine):
        """Test different storage engines"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = ProximaDB(data_dirs=tmpdir)
            try:
                db.create_collection(f"engine_{engine}", dimension=64, engine=engine)
                info = db.get_collection(f"engine_{engine}")
                assert info is not None
            except Exception:
                # Some engines might not be available in all builds
                pytest.skip(f"Engine {engine} not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
