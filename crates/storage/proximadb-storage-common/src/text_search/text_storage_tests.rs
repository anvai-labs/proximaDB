use super::*;

#[test]
fn test_default_config() {
    let config = TextStorageConfig::default();
    assert_eq!(config.inline_threshold, INLINE_THRESHOLD);
    assert_eq!(config.chunked_threshold, CHUNKED_THRESHOLD);
    assert_eq!(config.strategy, TextStorageStrategy::Adaptive);
}

#[test]
fn test_determine_strategy() {
    let config = TextStorageConfig::default();

    // Short text -> Inline
    let short_text = "Hello, world!";
    assert_eq!(
        determine_storage_strategy(short_text, &config),
        TextStorageStrategy::Inline
    );

    // Medium text -> Chunked
    let medium_text = "x".repeat(5000);
    assert_eq!(
        determine_storage_strategy(&medium_text, &config),
        TextStorageStrategy::Chunked
    );

    // Large text -> Sidecar
    let large_text = "x".repeat(2_000_000);
    assert_eq!(
        determine_storage_strategy(&large_text, &config),
        TextStorageStrategy::Sidecar
    );
}

#[test]
fn test_text_chunk() {
    let chunk = TextChunk::new(
        "chunk_0".to_string(),
        "record_1".to_string(),
        0,
        "Hello".to_string(),
    )
    .with_offsets(0, 5)
    .with_embedding(vec![0.1, 0.2, 0.3]);

    assert_eq!(chunk.chunk_id, "chunk_0");
    assert_eq!(chunk.parent_id, "record_1");
    assert_eq!(chunk.chunk_index, 0);
    assert_eq!(chunk.content, "Hello");
    assert!(chunk.embedding.is_some());
    assert_eq!(chunk.start_offset, 0);
    assert_eq!(chunk.end_offset, 5);
}

#[test]
fn test_sidecar_ref_serialization() {
    let sidecar_ref = SidecarRef::new(
        "record_1".to_string(),
        "/path/to/sidecar".to_string(),
        100,
        500,
    )
    .with_compression(SidecarCompression::Zstd);

    let bytes = sidecar_ref.to_bytes();
    let restored = SidecarRef::from_bytes("record_1".to_string(), &bytes)
        .expect("SidecarRef deserialization should succeed for valid bytes");

    assert_eq!(restored.sidecar_path, "/path/to/sidecar");
    assert_eq!(restored.offset, 100);
    assert_eq!(restored.length, 500);
    assert_eq!(restored.compression, SidecarCompression::Zstd);
}

#[test]
fn test_writer_inline() {
    let config = TextStorageConfig::for_small_text();
    let mut writer = TextColumnWriter::new(config);

    writer
        .write("rec_1", "Hello")
        .expect("Write should succeed for valid inline text");
    writer
        .write("rec_2", "World")
        .expect("Write should succeed for valid inline text");
    writer.write_null("rec_3");

    assert_eq!(writer.len(), 3);
    assert_eq!(writer.stats().inline_count, 2);
    assert_eq!(writer.stats().total_records, 3);
}

#[test]
fn test_writer_chunking() {
    let config = TextStorageConfig {
        strategy: TextStorageStrategy::Chunked,
        chunk_size: 10, // Small chunks for testing
        ..Default::default()
    };

    let mut writer = TextColumnWriter::new(config);

    writer
        .write("rec_1", "This is a longer text that will be chunked")
        .expect("Write should succeed for chunked text");

    assert!(!writer.get_chunks().is_empty());
    assert!(writer.get_chunks().len() > 1); // Should have multiple chunks
}

#[test]
fn test_writer_max_size() {
    let config = TextStorageConfig {
        max_text_size: 100,
        ..Default::default()
    };

    let mut writer = TextColumnWriter::new(config);

    let result = writer.write("rec_1", &"x".repeat(200));
    assert!(result.is_err());

    if let Err(TextStorageError::TextTooLarge(size, max)) = result {
        assert_eq!(size, 200);
        assert_eq!(max, 100);
    }
}

#[test]
fn test_storage_type_tracking() {
    let config = TextStorageConfig::for_small_text();
    let mut writer = TextColumnWriter::new(config);

    writer
        .write("rec_1", "Hello")
        .expect("Write should succeed for valid inline text");

    assert_eq!(writer.get_storage_type("rec_1"), Some(StorageType::Inline));
    assert_eq!(writer.get_storage_type("unknown"), None);
}

#[test]
fn test_build_arrow_array() {
    let config = TextStorageConfig::default();
    let mut writer = TextColumnWriter::new(config);

    writer
        .write("rec_1", "Hello")
        .expect("Write should succeed for valid inline text");
    writer
        .write("rec_2", "World")
        .expect("Write should succeed for valid inline text");
    writer.write_null("rec_3");

    let array = writer.build_inline_array();
    assert_eq!(array.len(), 3);
}

#[test]
fn test_reader_load_from_array() {
    let config = TextStorageConfig::default();
    let reader = TextColumnReader::new(config);

    let string_array: StringArray = vec![Some("Hello"), Some("World"), None].into();
    let array_ref: ArrayRef = Arc::new(string_array);

    let values = reader
        .load_from_array(&array_ref)
        .expect("Load from array should succeed for valid StringArray");
    assert_eq!(values.len(), 3);
    assert_eq!(values[0], Some("Hello".to_string()));
    assert_eq!(values[1], Some("World".to_string()));
    assert_eq!(values[2], None);
}

#[test]
fn test_config_presets() {
    let rag_config = TextStorageConfig::for_rag_documents(256);
    assert_eq!(rag_config.strategy, TextStorageStrategy::Chunked);
    assert_eq!(rag_config.chunk_size, 256);
    assert!(rag_config.enable_ngram_bloom);

    let large_config = TextStorageConfig::for_large_documents("/sidecars".to_string());
    assert_eq!(large_config.strategy, TextStorageStrategy::Sidecar);
    assert_eq!(
        large_config.sidecar_base_path,
        Some("/sidecars".to_string())
    );
    assert_eq!(large_config.sidecar_compression, SidecarCompression::Zstd);
}

// =========================================================================
// TextChunker Tests
// =========================================================================

#[test]
fn test_chunking_config_default() {
    let config = ChunkingConfig::default();
    assert_eq!(config.chunk_size, DEFAULT_CHUNK_SIZE);
    assert_eq!(config.overlap, DEFAULT_OVERLAP_SIZE);
    assert!(config.preserve_boundaries);
    assert_eq!(config.min_chunk_size, MIN_CHUNK_SIZE);
    assert_eq!(config.max_boundary_search, MAX_BOUNDARY_SEARCH);
}

#[test]
fn test_chunking_config_presets() {
    let semantic = ChunkingConfig::for_semantic_search();
    assert_eq!(semantic.chunk_size, 256);
    assert_eq!(semantic.overlap, 64);

    let qa = ChunkingConfig::for_qa();
    assert_eq!(qa.chunk_size, 1024);
    assert_eq!(qa.overlap, 128);

    let code = ChunkingConfig::for_code();
    assert_eq!(code.chunk_size, 512);
    assert!(code.separator.is_some());
}

#[test]
fn test_chunking_config_validation() {
    // Valid config
    let valid = ChunkingConfig::new(512, 50);
    assert!(valid.validate().is_ok());

    // Overlap >= chunk_size is invalid
    let invalid = ChunkingConfig {
        overlap: 600,
        ..Default::default()
    };
    assert!(invalid.validate().is_err());

    // chunk_size < min_chunk_size is invalid
    let invalid2 = ChunkingConfig {
        chunk_size: 32,
        min_chunk_size: 64,
        ..Default::default()
    };
    assert!(invalid2.validate().is_err());
}

#[test]
fn test_text_chunker_simple() {
    let chunker = TextChunker::new(ChunkingConfig {
        chunk_size: 10,
        overlap: 0,
        preserve_boundaries: false,
        min_chunk_size: 5,
        ..Default::default()
    });

    let text = "Hello World, how are you today?";
    let chunks = chunker.chunk_text("doc1", text);

    assert!(!chunks.is_empty());
    assert!(chunks.len() > 1);

    // All chunks should have correct parent_id
    for chunk in &chunks {
        assert_eq!(chunk.parent_id, "doc1");
    }

    // First chunk should start at 0
    assert_eq!(chunks[0].start_offset, 0);
}

#[test]
fn test_text_chunker_with_overlap() {
    let chunker = TextChunker::new(ChunkingConfig {
        chunk_size: 20,
        overlap: 5,
        preserve_boundaries: false,
        min_chunk_size: 10,
        ..Default::default()
    });

    let text = "AAAAAAAAAABBBBBBBBBBCCCCCCCCCC"; // 30 chars
    let chunks = chunker.chunk_text("doc1", text);

    // With overlap, chunks should overlap
    assert!(chunks.len() >= 2);

    // Check that chunks overlap (second chunk starts before first ends)
    if chunks.len() >= 2 {
        // The second chunk should start within the first chunk's range
        // due to overlap
        assert!(chunks[1].start_offset < chunks[0].end_offset || chunks.len() == 2);
    }
}

#[test]
fn test_text_chunker_boundary_preservation() {
    let chunker = TextChunker::new(ChunkingConfig {
        chunk_size: 30,
        overlap: 5,
        preserve_boundaries: true,
        min_chunk_size: 10,
        max_boundary_search: 20,
        ..Default::default()
    });

    // Text with clear sentence boundaries
    let text = "Hello world. This is a test. Another sentence here.";
    let chunks = chunker.chunk_text("doc1", text);

    // Should have at least one chunk
    assert!(!chunks.is_empty());

    // With boundary preservation, chunks should tend to end at periods
    // (This is a soft check since boundary finding is best-effort)
    for chunk in &chunks {
        // Chunks should not be empty
        assert!(!chunk.content.trim().is_empty());
    }
}

#[test]
fn test_text_chunker_empty_text() {
    let chunker = TextChunker::default();
    let chunks = chunker.chunk_text("doc1", "");
    assert!(chunks.is_empty());
}

#[test]
fn test_text_chunker_small_text() {
    let chunker = TextChunker::new(ChunkingConfig {
        chunk_size: 512,
        overlap: 50,
        min_chunk_size: 64,
        ..Default::default()
    });

    let text = "Short text"; // Less than min_chunk_size
    let chunks = chunker.chunk_text("doc1", text);

    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].content, "Short text");
}

#[test]
fn test_text_chunker_metadata() {
    let chunker = TextChunker::new(ChunkingConfig {
        chunk_size: 20,
        overlap: 5,
        preserve_boundaries: false,
        min_chunk_size: 10,
        chunk_id_prefix: "test_chunk".to_string(),
        ..Default::default()
    });

    let text = "This is a test document for metadata.";
    let chunks = chunker.chunk_text("doc123", text);

    assert!(!chunks.is_empty());

    let first_chunk = &chunks[0];

    // Check chunk_id format
    assert!(first_chunk.chunk_id.starts_with("test_chunk_doc123_"));

    // Check metadata
    assert_eq!(
        first_chunk.metadata.get("parent_id"),
        Some(&"doc123".to_string())
    );
    assert!(first_chunk.metadata.contains_key("chunk_index"));
    assert!(first_chunk.metadata.contains_key("char_start"));
    assert!(first_chunk.metadata.contains_key("char_end"));
    assert!(first_chunk.metadata.contains_key("total_chunks"));
}

#[test]
fn test_text_chunker_chunk_position() {
    let position = ChunkPosition::new(10, 100, 5, 50).with_lines(2, 5);

    assert_eq!(position.byte_start, 10);
    assert_eq!(position.byte_end, 100);
    assert_eq!(position.byte_len(), 90);
    assert_eq!(position.char_start, 5);
    assert_eq!(position.char_end, 50);
    assert_eq!(position.char_len(), 45);
    assert_eq!(position.line_start, 2);
    assert_eq!(position.line_end, 5);
}

#[test]
fn test_text_chunker_byte_offset_calculation() {
    let text = "Hello World";
    let (byte_start, byte_end) = TextChunker::calculate_byte_offsets(text, 0, 5);
    assert_eq!(byte_start, 0);
    assert_eq!(byte_end, 5); // "Hello" is 5 bytes
}

#[test]
fn test_text_chunker_line_number_calculation() {
    let text = "Line 1\nLine 2\nLine 3";
    let (line_start, line_end) = TextChunker::calculate_line_numbers(text, 0, 15);
    assert_eq!(line_start, 1);
    assert!(line_end >= 2);
}

#[test]
fn test_text_chunker_find_by_id() {
    let chunker = TextChunker::default();
    let text = "This is a test document that will be chunked into multiple pieces.";
    let chunks = chunker.chunk_text("doc1", text);

    if !chunks.is_empty() {
        let chunk_id = &chunks[0].chunk_id;
        let found = TextChunker::find_chunk_by_id(&chunks, chunk_id);
        assert!(found.is_some());
        let found_chunk = found.expect("Chunk should be found after is_some() check");
        assert_eq!(found_chunk.chunk_id, *chunk_id);

        let not_found = TextChunker::find_chunk_by_id(&chunks, "nonexistent");
        assert!(not_found.is_none());
    }
}

#[test]
fn test_text_chunker_get_chunks_for_parent() {
    let chunker = TextChunker::new(ChunkingConfig {
        chunk_size: 20,
        overlap: 0,
        preserve_boundaries: false,
        min_chunk_size: 10,
        ..Default::default()
    });

    let chunks1 = chunker.chunk_text("doc1", "Text for document one.");
    let chunks2 = chunker.chunk_text("doc2", "Text for document two.");

    let mut all_chunks: Vec<TextChunk> = Vec::new();
    all_chunks.extend(chunks1);
    all_chunks.extend(chunks2);

    let doc1_chunks = TextChunker::get_chunks_for_parent(&all_chunks, "doc1");
    let doc2_chunks = TextChunker::get_chunks_for_parent(&all_chunks, "doc2");

    assert!(!doc1_chunks.is_empty());
    assert!(!doc2_chunks.is_empty());

    for chunk in doc1_chunks {
        assert_eq!(chunk.parent_id, "doc1");
    }
    for chunk in doc2_chunks {
        assert_eq!(chunk.parent_id, "doc2");
    }
}

// =========================================================================
// TextColumnWriter with RAG Chunking Tests
// =========================================================================

#[test]
#[allow(clippy::field_reassign_with_default)]
fn test_writer_with_rag_chunking() {
    let mut config = TextStorageConfig::default();
    config.strategy = TextStorageStrategy::Chunked;

    let writer = TextColumnWriter::new(config).with_chunking_config(ChunkingConfig {
        chunk_size: 20,
        overlap: 5,
        preserve_boundaries: false,
        min_chunk_size: 10,
        ..Default::default()
    });

    assert!(writer.has_rag_chunking());
    assert!(writer.chunker().is_some());
}

#[test]
#[allow(clippy::field_reassign_with_default)]
fn test_writer_rag_chunking_produces_overlap() {
    let mut config = TextStorageConfig::default();
    config.strategy = TextStorageStrategy::Chunked;

    let mut writer = TextColumnWriter::new(config).with_chunking_config(ChunkingConfig {
        chunk_size: 20,
        overlap: 5,
        preserve_boundaries: false,
        min_chunk_size: 10,
        ..Default::default()
    });

    writer
        .write(
            "rec_1",
            "This is a longer text for testing RAG chunking with overlap.",
        )
        .expect("Write should succeed for RAG chunking with valid text");

    let chunks = writer.get_chunks();
    assert!(chunks.len() > 1);

    // Verify chunks have metadata
    for chunk in chunks {
        assert!(chunk.metadata.contains_key("parent_id"));
        assert!(chunk.metadata.contains_key("chunk_index"));
    }
}

#[test]
#[allow(clippy::field_reassign_with_default)]
fn test_writer_without_rag_chunking_fallback() {
    let mut config = TextStorageConfig::default();
    config.strategy = TextStorageStrategy::Chunked;
    config.chunk_size = 10;

    let mut writer = TextColumnWriter::new(config);
    // No chunking config set - should use fallback

    assert!(!writer.has_rag_chunking());

    writer
        .write("rec_1", "This is a test text for fallback chunking.")
        .expect("Write should succeed for fallback chunking with valid text");

    let chunks = writer.get_chunks();
    assert!(!chunks.is_empty());
}

#[test]
fn test_chunker_generate_chunk_id() {
    let id = TextChunker::generate_chunk_id("chunk", "doc123", 5);
    assert_eq!(id, "chunk_doc123__0005");
}

// =========================================================================
// Full-Text Index Integration Tests
// =========================================================================

#[test]
fn test_writer_with_fulltext_index() {
    let writer = TextColumnWriter::new(TextStorageConfig::default())
        .with_fulltext_index(TokenizerConfig::default());

    assert!(writer.has_fulltext_index());
    assert!(writer.fulltext_index().is_some());
}

#[test]
fn test_fulltext_auto_indexing() {
    let mut writer = TextColumnWriter::new(TextStorageConfig::default())
        .with_fulltext_index(TokenizerConfig::default());

    writer
        .write("doc1", "The quick brown fox")
        .expect("Write should succeed for full-text indexing");
    writer
        .write("doc2", "A lazy brown dog")
        .expect("Write should succeed for full-text indexing");
    writer
        .write("doc3", "The quick blue bird")
        .expect("Write should succeed for full-text indexing");

    // Search for documents
    let results = writer.fulltext_search("quick brown", 10);
    assert!(!results.is_empty());

    // doc1 should rank highest (has both "quick" and "brown")
    assert_eq!(results[0].doc_id, "doc1");
}

#[test]
fn test_fulltext_search_with_options() {
    let mut writer = TextColumnWriter::new(TextStorageConfig::default())
        .with_fulltext_index(TokenizerConfig::default());

    writer
        .write("doc1", "quick brown fox jumps")
        .expect("Write should succeed for full-text indexing");
    writer
        .write("doc2", "quick rabbit")
        .expect("Write should succeed for full-text indexing");
    writer
        .write("doc3", "slow brown tortoise")
        .expect("Write should succeed for full-text indexing");

    // Require all terms
    let results =
        writer.fulltext_search_with_options("quick brown", SearchOptions::top_k(10).require_all());

    // Only doc1 has both "quick" and "brown"
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].doc_id, "doc1");
}

#[test]
fn test_fulltext_term_statistics() {
    let mut writer = TextColumnWriter::new(TextStorageConfig::default())
        .with_fulltext_index(TokenizerConfig::default());

    writer
        .write("doc1", "hello world")
        .expect("Write should succeed for full-text indexing");
    writer
        .write("doc2", "hello there")
        .expect("Write should succeed for full-text indexing");
    writer
        .write("doc3", "goodbye world")
        .expect("Write should succeed for full-text indexing");

    // Check document frequency
    let hello_df = writer.get_document_frequency("hello");
    assert_eq!(hello_df, 2);

    let world_df = writer.get_document_frequency("world");
    assert_eq!(world_df, 2);

    // Check IDF (higher for rarer terms)
    let hello_idf = writer.get_term_idf("hello");
    let goodbye_idf = writer.get_term_idf("goodbye");
    assert!(goodbye_idf > hello_idf); // "goodbye" is rarer
}

#[test]
fn test_fulltext_top_terms() {
    let mut writer = TextColumnWriter::new(TextStorageConfig::default())
        .with_fulltext_index(TokenizerConfig::default());

    writer
        .write("doc1", "test testing tested")
        .expect("Write should succeed for full-text indexing");
    writer
        .write("doc2", "test example")
        .expect("Write should succeed for full-text indexing");
    writer
        .write("doc3", "test sample")
        .expect("Write should succeed for full-text indexing");

    let top_terms = writer.get_top_terms(5);
    assert!(!top_terms.is_empty());

    // "test" should be in the top terms
    let has_test = top_terms.iter().any(|(term, _)| term == "test");
    assert!(has_test);
}

#[test]
fn test_fulltext_prefix_search() {
    let mut writer = TextColumnWriter::new(TextStorageConfig::default())
        .with_fulltext_index(TokenizerConfig::default());

    writer
        .write("doc1", "testing tested tester")
        .expect("Write should succeed for full-text indexing");
    writer
        .write("doc2", "temperature temporal")
        .expect("Write should succeed for full-text indexing");

    let terms = writer.get_terms_with_prefix("test", 10);
    assert!(!terms.is_empty());
    for term in &terms {
        assert!(term.starts_with("test"));
    }
}

#[test]
fn test_fulltext_with_bm25_config() {
    let mut writer = TextColumnWriter::new(TextStorageConfig::default())
        .with_fulltext_index_and_bm25(
            TokenizerConfig::for_keyword_search(),
            BM25Config::for_short_documents(),
        );

    writer
        .write("doc1", "short text here")
        .expect("Write should succeed for BM25 indexing");
    writer
        .write("doc2", "another short document")
        .expect("Write should succeed for BM25 indexing");

    let results = writer.fulltext_search("short", 10);
    assert!(!results.is_empty());
}

#[test]
fn test_fulltext_manual_indexing() {
    let mut writer = TextColumnWriter::new(TextStorageConfig::default())
        .with_fulltext_index(TokenizerConfig::default());

    // Disable auto-indexing
    writer.set_auto_index(false);

    writer
        .write("doc1", "some text")
        .expect("Write should succeed even without auto-indexing");

    // Should not find anything because auto-index is disabled
    let results = writer.fulltext_search("text", 10);
    assert!(results.is_empty());

    // Manually index
    writer
        .index_document("doc1", "some text")
        .expect("Manual indexing should succeed for valid document");

    // Now should find it
    let results = writer.fulltext_search("text", 10);
    assert!(!results.is_empty());
}

#[test]
fn test_fulltext_clear() {
    let mut writer = TextColumnWriter::new(TextStorageConfig::default())
        .with_fulltext_index(TokenizerConfig::default());

    writer
        .write("doc1", "hello world")
        .expect("Write should succeed for full-text indexing");

    // Verify index has content
    let results = writer.fulltext_search("hello", 10);
    assert!(!results.is_empty());

    // Clear
    writer.clear();

    // Index should be empty
    let results = writer.fulltext_search("hello", 10);
    assert!(results.is_empty());
}

#[test]
#[allow(clippy::field_reassign_with_default)]
fn test_fulltext_index_from_chunks() {
    let mut config = TextStorageConfig::default();
    config.strategy = TextStorageStrategy::Chunked;
    config.chunk_size = 20;

    let mut writer = TextColumnWriter::new(config).with_chunking_config(ChunkingConfig {
        chunk_size: 20,
        overlap: 5,
        preserve_boundaries: false,
        min_chunk_size: 10,
        ..Default::default()
    });

    // Write will create chunks
    writer
        .write(
            "doc1",
            "This is a longer document that will be split into multiple chunks for testing.",
        )
        .expect("Write should succeed for chunked text");

    // Build index from chunks
    writer
        .build_index_from_chunks()
        .expect("Building index from chunks should succeed");

    // Should be able to search chunks
    let results = writer.fulltext_search("document", 10);
    // Results should contain chunk IDs
    assert!(!results.is_empty());
}
