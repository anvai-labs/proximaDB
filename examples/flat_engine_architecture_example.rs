//! Flat Storage Engine Architecture Example
//!
//! This example demonstrates the new flat storage engine structure
//! where all 12 engines are accessible directly at the engines/ level.

use proximadb::storage::engines::{StorageEngineFactory, WorkloadType};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🏗️  ProximaDB Flat Storage Engine Architecture Example\n");

    // 1. Demonstrate direct engine access (NEW flat structure)
    demonstrate_direct_engine_access()?;

    // 2. Demonstrate automatic engine selection
    demonstrate_automatic_selection().await?;

    // 3. Demonstrate all 12 engines
    demonstrate_all_engines().await?;

    Ok(())
}

/// Demonstrates the NEW flat import structure (Phase 2 complete)
fn demonstrate_direct_engine_access() -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 1. Direct Engine Access (Flat Structure)");
    println!("   ─────────────────────────────────────");

    println!("   ✅ All 12 storage engines now accessible at same level:");
    println!();

    // Major Engines (Phase 1)
    println!("   🏭 Major Engines (Phase 1):");
    println!("      use proximadb::storage::engines::sst::SstEngine;");
    println!("      use proximadb::storage::engines::nova::NovaEngine;");
    println!("      use proximadb::storage::engines::nova::NovaEngine;");
    println!("      use proximadb::storage::engines::swift::SwiftEngine;");
    println!("      use proximadb::storage::engines::raptor::RaptorEngine;");
    println!("      use proximadb::storage::engines::helix::HelixEngine;");
    println!();

    // Specialized Engines (Phase 2)
    println!("   🔧 Specialized Engines (Phase 2):");
    println!("      use proximadb::storage::engines::cedar::CedarEngine;");
    println!("      use proximadb::storage::engines::chrono::ChronoEngine;");
    println!("      use proximadb::storage::engines::eventlog::EventLogEngine;");
    println!("      use proximadb::storage::engines::sequoia::SequoiaEngine;");
    println!("      use proximadb::storage::engines::titan::TitanEngine;");
    println!("      use proximadb::storage::engines::tst::TimeSeriesEngine;");
    println!();

    println!("   📈 Import Path Improvement:");
    println!("      Before: crate::storage::engines::impls::sst::SstEngine");
    println!("      After:  crate::storage::engines::sst::SstEngine");
    println!("      Reduction: 5 segments → 4 segments (20% improvement)");

    println!();
    Ok(())
}

/// Demonstrates automatic engine selection based on workload
async fn demonstrate_automatic_selection() -> Result<(), Box<dyn std::error::Error>> {
    println!("🤖 2. Automatic Engine Selection");
    println!("   ─────────────────────────────");

    println!("   ✅ StorageEngineFactory automatically selects optimal engine:");
    println!();

    // Transactional workload → SST engine
    let _oltp_engine = StorageEngineFactory::create_for_workload(WorkloadType::Transactional)?;
    println!("   📊 Transactional Workload → SST Engine (Real-time queries, frequent updates)");

    // Analytics workload → NOVA engine
    let _olap_engine = StorageEngineFactory::create_for_workload(WorkloadType::Analytics)?;
    println!("   📈 Analytics Workload → NOVA Engine (Analytics, batch operations)");

    // Mixed workload → NOVA engine
    let _mixed_engine = StorageEngineFactory::create_for_workload(WorkloadType::Mixed)?;
    println!("   🔀 Mixed Workload → NOVA Engine (Combined OLTP/OLAP)");

    println!("\n   💡 Factory pattern remains unchanged - backward compatible!");

    println!();
    Ok(())
}

/// Demonstrates all 12 engines with their use cases
async fn demonstrate_all_engines() -> Result<(), Box<dyn std::error::Error>> {
    println!("🏭️ 3. All 12 Storage Engines Overview");
    println!("   ───────────────────────────────");

    let engines = vec![
        (
            "SST",
            "Real-time queries, frequent updates",
            "Transactional",
        ),
        ("NOVA", "Analytics, batch operations", "Analytics"),
        ("NOVA", "Mixed workloads", "Mixed"),
        ("SWIFT", "High-throughput", "Experimental"),
        ("RAPTOR", "Matrix operations", "Analytics"),
        ("HELIX", "Spatial queries, range scans", "Spatial"),
        ("CEDAR", "JSON document CRUD", "Documents"),
        ("CHRONO", "Metrics, logs, traces", "Observability"),
        ("EventLog", "Audit trails, event replay", "Audit"),
        ("SEQUOIA", "Relational data", "Relational"),
        ("TITAN", "Graph traversals", "Graph"),
        ("TST", "Time-series data", "Time-series"),
    ];

    println!("   ✅ Complete Engine Portfolio:");
    println!();

    for (i, (engine, description, workload)) in engines.iter().enumerate() {
        println!("   {}. **{}** - {}", i + 1, engine, description);
        println!("      Best for: {}", workload);
        println!(
            "      Import: use proximadb::storage::engines::{}::{}Engine;",
            engine.to_lowercase(),
            engine
        );
        println!();
    }

    println!("   🎯 Key Benefits:");
    println!("      ✅ Consistent 4-segment import paths");
    println!("      ✅ All engines at same level");
    println!("      ✅ No nested impls:: namespace");
    println!("      ✅ Improved discoverability");
    println!("      ✅ Better maintainability");

    println!();
    Ok(())
}
