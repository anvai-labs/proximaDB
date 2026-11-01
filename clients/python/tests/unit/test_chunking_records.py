"""Chunking record-native helper tests."""

from proximadb_sdk.chunking import (
    TextChunk,
    create_records,
    create_vector_records,
    prepare_records,
    prepare_vector_records,
)
from proximadb_sdk.models import VectorRecord


def test_create_records_returns_proximarecord_payloads():
    chunk = TextChunk(
        text="hello world",
        chunk_id="doc_0",
        start_pos=0,
        end_pos=11,
        metadata={"source_id": "doc", "chunk_index": 0},
    )

    records = create_records([chunk], [[0.1, 0.2]], {"kind": "note"})

    assert records == [
        {
            "id": "doc_0",
            "vector": [0.1, 0.2],
            "props": {
                "kind": "note",
                "source_id": "doc",
                "chunk_index": 0,
                "chunk_length": 11,
                "chunk_id": "doc_0",
                "text_preview": "hello world",
                "embedding_dimension": 2,
            },
            # TD-REC-1: no top-level source; text rides in text_fields.
            "text_fields": [{"name": "chunk_text", "content": "hello world"}],
        }
    ]


def test_create_vector_records_is_compatibility_wrapper():
    chunk = TextChunk(
        text="hello world",
        chunk_id="doc_0",
        start_pos=0,
        end_pos=11,
        metadata={"source_id": "doc"},
    )

    records = create_vector_records([chunk], [[0.1, 0.2]])

    assert isinstance(records[0], VectorRecord)
    assert records[0].id == "doc_0"
    assert records[0].metadata["text_preview"] == "hello world"


def test_prepare_records_returns_props_and_source_text():
    records = prepare_records(
        {
            "chunks": [{"id": "c1", "text": "Product", "embedding": [0.1]}],
            "model": "test-model",
        },
        source_id="doc1",
        source_type="catalog",
    )

    assert records[0]["id"] == "c1"
    assert records[0]["vector"] == [0.1]
    assert records[0]["props"]["source_id"] == "doc1"
    assert records[0]["props"]["source_type"] == "catalog"
    assert records[0]["props"]["embedding_model"] == "test-model"
    # TD-REC-1: no top-level source; the content is in text_fields.
    assert records[0]["text_fields"][0]["content"] == "Product"


def test_prepare_vector_records_is_compatibility_wrapper():
    records = prepare_vector_records(
        {"chunks": [{"id": "c1", "text": "Product", "embedding": [0.1]}]},
        source_id="doc1",
    )

    assert isinstance(records[0], VectorRecord)
    assert records[0].metadata["source_id"] == "doc1"
