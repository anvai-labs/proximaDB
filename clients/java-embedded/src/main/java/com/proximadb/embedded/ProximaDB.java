/*
 * Copyright 2025 Vijaykumar Singh
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */
package com.proximadb.embedded;

import java.io.Closeable;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.List;

/**
 * ProximaDB Embedded - In-process vector database for Java.
 *
 * <p>This class provides direct in-process access to ProximaDB's high-performance
 * Rust core without any network overhead. Perfect for applications that need
 * fast, local vector storage.
 *
 * <h2>Features</h2>
 * <ul>
 *   <li>Zero network overhead - direct JNI calls to Rust core</li>
 *   <li>Multi-disk support with weighted distribution</li>
 *   <li>SIMD-accelerated vector operations</li>
 *   <li>Full WAL persistence and crash recovery</li>
 * </ul>
 *
 * <h2>Example Usage</h2>
 * <pre>{@code
 * try (ProximaDB db = ProximaDB.open("./my_database")) {
 *     // Create collection
 *     db.createCollection("embeddings", 768);
 *
 *     // Insert vectors
 *     String[] ids = {"vec_0", "vec_1"};
 *     float[][] vectors = {{0.1f, 0.2f, ...}, {0.3f, 0.4f, ...}};
 *     db.insert("embeddings", ids, vectors);
 *
 *     // Search
 *     float[] query = {0.1f, 0.2f, ...};
 *     SearchResult[] results = db.search("embeddings", query, 10);
 * }
 * }</pre>
 */
public class ProximaDB implements Closeable, AutoCloseable {

    // Native library handle
    private long nativeHandle = 0;

    // Load native library on class load
    static {
        loadNativeLibrary();
    }

    private static void loadNativeLibrary() {
        try {
            // Try loading from java.library.path first
            System.loadLibrary("proximadb_java");
        } catch (UnsatisfiedLinkError e1) {
            // Try extracting from JAR
            try {
                loadLibraryFromJar();
            } catch (IOException e2) {
                throw new RuntimeException(
                    "Failed to load native library. Ensure libproximadb_java is in java.library.path " +
                    "or bundled in the JAR. Original error: " + e1.getMessage(), e2);
            }
        }
    }

    private static void loadLibraryFromJar() throws IOException {
        String os = System.getProperty("os.name").toLowerCase();
        String arch = System.getProperty("os.arch").toLowerCase();

        String libName;
        if (os.contains("linux")) {
            libName = "libproximadb_java.so";
        } else if (os.contains("mac")) {
            libName = "libproximadb_java.dylib";
        } else if (os.contains("windows")) {
            libName = "proximadb_java.dll";
        } else {
            throw new IOException("Unsupported OS: " + os);
        }

        String resourcePath = "/native/" + os + "/" + arch + "/" + libName;

        try (InputStream is = ProximaDB.class.getResourceAsStream(resourcePath)) {
            if (is == null) {
                throw new IOException("Native library not found in JAR: " + resourcePath);
            }

            Path tempDir = Files.createTempDirectory("proximadb");
            Path tempLib = tempDir.resolve(libName);
            Files.copy(is, tempLib, StandardCopyOption.REPLACE_EXISTING);

            System.load(tempLib.toString());

            // Schedule cleanup on shutdown
            tempLib.toFile().deleteOnExit();
            tempDir.toFile().deleteOnExit();
        }
    }

    // Native methods
    private native void nativeCreate(String dataDir, String metadataDir, int cacheSizeMb, String engine);
    private native void nativeCreateMultiDisk(String[] diskPaths, float[] diskWeights, String metadataDir, int cacheSizeMb, String engine);
    private native void nativeClose();
    private native void nativeCreateCollection(String name, int dimension, String engine);
    private native void nativeDeleteCollection(String name);
    private native int nativeInsert(String collection, String[] ids, float[][] vectors);
    private native SearchResult[] nativeSearch(String collection, float[] query, int topK);
    private native void nativeFlush();

    /**
     * Create a new ProximaDB instance with default settings.
     *
     * @param dataDir Path to the data directory
     * @return New ProximaDB instance
     */
    public static ProximaDB open(String dataDir) {
        return new Builder().dataDir(dataDir).build();
    }

    /**
     * Create a new ProximaDB instance with custom configuration.
     *
     * @return Builder for configuring the database
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Builder for configuring ProximaDB instances.
     */
    public static class Builder {
        private String dataDir = "./data";
        private String metadataDir = null;
        private int cacheSizeMb = 512;
        private String defaultEngine = "sst";
        private List<DiskConfig> disks = null;

        /**
         * Set the data directory path.
         */
        public Builder dataDir(String path) {
            this.dataDir = path;
            return this;
        }

        /**
         * Set the metadata directory path.
         * If not set, defaults to dataDir/metadata.
         */
        public Builder metadataDir(String path) {
            this.metadataDir = path;
            return this;
        }

        /**
         * Set the cache size in megabytes.
         */
        public Builder cacheSizeMb(int size) {
            this.cacheSizeMb = size;
            return this;
        }

        /**
         * Set the default storage engine.
         * Options: "sst", "nova", "swift", "raptor", "helix"
         */
        public Builder defaultEngine(String engine) {
            this.defaultEngine = engine;
            return this;
        }

        /**
         * Configure multi-disk storage.
         * When set, dataDir is ignored in favor of disk configurations.
         */
        public Builder disks(List<DiskConfig> disks) {
            this.disks = disks;
            return this;
        }

        /**
         * Build the ProximaDB instance.
         */
        public ProximaDB build() {
            ProximaDB db = new ProximaDB();

            if (disks != null && !disks.isEmpty()) {
                String[] paths = disks.stream().map(d -> d.getPath()).toArray(String[]::new);
                float[] weights = new float[disks.size()];
                for (int i = 0; i < disks.size(); i++) {
                    weights[i] = disks.get(i).getWeight();
                }
                db.nativeCreateMultiDisk(paths, weights, metadataDir, cacheSizeMb, defaultEngine);
            } else {
                db.nativeCreate(dataDir, metadataDir, cacheSizeMb, defaultEngine);
            }

            return db;
        }
    }

    // Private constructor - use open() or builder()
    private ProximaDB() {}

    /**
     * Create a new collection.
     *
     * @param name Collection name
     * @param dimension Vector dimension
     */
    public void createCollection(String name, int dimension) {
        createCollection(name, dimension, null);
    }

    /**
     * Create a new collection with a specific storage engine.
     *
     * @param name Collection name
     * @param dimension Vector dimension
     * @param engine Storage engine type (null for default)
     */
    public void createCollection(String name, int dimension, String engine) {
        nativeCreateCollection(name, dimension, engine);
    }

    /**
     * Delete a collection.
     *
     * @param name Collection name to delete
     */
    public void deleteCollection(String name) {
        nativeDeleteCollection(name);
    }

    /**
     * Insert vectors into a collection.
     *
     * @param collection Collection name
     * @param ids Vector IDs
     * @param vectors 2D array of vectors [count][dimension]
     * @return Number of vectors inserted
     */
    public int insert(String collection, String[] ids, float[][] vectors) {
        if (ids.length != vectors.length) {
            throw new IllegalArgumentException("ids and vectors must have same length");
        }
        return nativeInsert(collection, ids, vectors);
    }

    /**
     * Search for similar vectors.
     *
     * @param collection Collection name
     * @param query Query vector
     * @param topK Number of results to return
     * @return Array of search results
     */
    public SearchResult[] search(String collection, float[] query, int topK) {
        return nativeSearch(collection, query, topK);
    }

    /**
     * Search for similar vectors with default top_k=10.
     */
    public SearchResult[] search(String collection, float[] query) {
        return search(collection, query, 10);
    }

    /**
     * Flush all pending writes to disk.
     */
    public void flush() {
        nativeFlush();
    }

    /**
     * Close the database and release resources.
     */
    @Override
    public void close() {
        if (nativeHandle != 0) {
            nativeClose();
            nativeHandle = 0;
        }
    }

    @Override
    protected void finalize() throws Throwable {
        try {
            close();
        } finally {
            super.finalize();
        }
    }
}
