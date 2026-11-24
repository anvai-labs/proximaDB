# Production Recovery Strategy: DirectToStorage vs ViaMemtable

## Current Issue

**ViaMemtable is a workaround, not a production solution.**

### Why ViaMemtable is Problematic for Production

1. **Memory Overflow Risk**
   - Large WAL files → all vectors in memory
   - Could exceed available RAM during recovery
   - No deterministic flush until later

2. **Upgrade Challenges**
   - Data sits in memtable after recovery
   - Unclear when it gets flushed to storage
   - Upgrade rollback would lose unflushed data

3. **Non-Deterministic Flush**
   - Memtable flush triggers are probabilistic
   - Can't guarantee when data hits disk
   - Makes recovery timing unpredictable

## Correct Solution: DirectToStorage

**DirectToStorage is the right approach:**

1. ✅ Flushes directly to persistent storage during recovery
2. ✅ Deterministic - data on disk immediately after recovery
3. ✅ Memory bounded - doesn't load all vectors
4. ✅ Survives immediate restart after recovery
5. ✅ Proper for production/upgrades

### The Problem We Need to Fix

**DirectToStorage requires storage engines to be registered BEFORE recovery runs.**

Current startup order (WRONG):
```
1. Start storage engine
2. Recover from WAL → calls DirectToStorage
   ❌ No storage engines registered yet!
   ❌ Skips all collections
3. Load collections (which would register engines)
   ❌ Too late, recovery already ran
```

Correct startup order (NEEDED):
```
1. Start storage engine
2. Load collection metadata
3. Pre-register storage engine stubs for each collection
4. Recover from WAL → DirectToStorage
   ✅ Engines registered!
   ✅ Can flush to storage
5. Complete collection initialization
```

## Implementation Plan

### Step 1: Pre-Register Storage Engines

**File:** `src/storage/engine.rs`

```rust
pub async fn start(&mut self) -> Result<()> {
    info!("🚀 STORAGE_ENGINE: Starting storage engine");

    // NEW: Load collections first to know what exists
    self.load_collections().await?;

    // NEW: Pre-register storage engines for all collections
    self.pre_register_engines_for_recovery().await?;

    // NOW: Recover from WAL with engines available
    self.recover_from_wal().await?;

    // Continue with rest of startup...
}

async fn pre_register_engines_for_recovery(&mut self) -> Result<()> {
    let collections = self.get_loaded_collections();

    // Get recovery manager
    let recovery_manager = self.write_ahead_log_manager
        .get_recovery_manager().await?;

    for collection in collections {
        // Register a placeholder/stub engine
        // Actual engine will be fully initialized later
        let stub_engine = self.create_engine_stub(&collection)?;
        recovery_manager.register_storage_engine(
            &collection.id,
            stub_engine
        ).await;
    }

    info!("Pre-registered {} storage engines for recovery", collections.len());
    Ok(())
}
```

### Step 2: Change RecoveryMode Back

**File:** `src/storage/persistence/write_ahead_log/recovery_manager.rs`

```rust
// After pre-registration is implemented:
recovery_mode: RecoveryMode::DirectToStorage,  // Now safe!
```

### Step 3: Implement Engine Stub

**File:** `src/storage/engine.rs`

```rust
fn create_engine_stub(&self, collection: &Collection) -> Arc<dyn UnifiedStorageEngine> {
    // Create minimal engine that can accept recovered vectors
    // Will be replaced with full engine later
    match collection.storage_engine {
        StorageEngine::SST => create_sst_stub(),
        StorageEngine::NOVA => create_nova_stub(),
        // etc.
    }
}
```

## Hybrid Approach (Recommended)

**Use BOTH modes intelligently:**

```rust
pub enum RecoveryMode {
    DirectToStorage,    // When engines available
    ViaMemtable,       // Fallback when engines not ready
    Hybrid,            // NEW: Try DirectToStorage, fallback to ViaMemtable
}

async fn recover_collection_internal(...) -> Result<(u64, u64)> {
    match recovery_mode {
        RecoveryMode::DirectToStorage => {
            if storage_engines.contains_key(collection_id) {
                // Direct flush to storage
                flush_to_storage(vectors).await?;
            } else {
                return Err("No storage engine");
            }
        }
        RecoveryMode::ViaMemtable => {
            // Recover to memtable
            recover_to_memtable(vectors).await?;
        }
        RecoveryMode::Hybrid => {
            // Try DirectToStorage first
            if storage_engines.contains_key(collection_id) {
                flush_to_storage(vectors).await?;
            } else {
                // Fallback to memtable
                warn!("No storage engine for {}, using memtable", collection_id);
                recover_to_memtable(vectors).await?;
            }
        }
    }
}
```

## Testing After Fix

Once DirectToStorage is working:

```bash
# Clean data
rm -rf /tmp/proximadb

# Run recovery test
cd clients/python
export PYTHONPATH=src
python3 tests/server_lifecycle/test_wal_persistence_detailed.py

# Should see in logs:
# ✅ Storage engine registered for collection
# ✅ Flushing to storage (not memtable)
# ✅ 20/20 vectors recovered to persistent storage
```

## Benefits of Proper DirectToStorage

1. **Deterministic Recovery**
   - Know exactly when data is persisted
   - Can verify storage files exist
   - Safe for immediate restart

2. **Memory Bounded**
   - Don't load all vectors into RAM
   - Stream through WAL files
   - Flush incrementally

3. **Production Ready**
   - Survives upgrades
   - Proper for Kubernetes
   - Clear recovery semantics

4. **Performance**
   - Can use storage engine optimizations
   - Batch writes to storage
   - Parallel recovery across collections

## Migration Path

**Phase 1 (Current):**
- ViaMemtable works for recovery
- Data accessible after restart
- Temporary solution

**Phase 2 (Recommended):**
- Implement pre-registration
- Switch to DirectToStorage
- Remove ViaMemtable dependency

**Phase 3 (Future):**
- Add Hybrid mode for robustness
- Fallback chain: Direct → Memtable → Error
- Maximum recovery success rate
