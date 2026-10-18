// SPDX-License-Identifier: Apache-2.0
//! Exercise the PUBLISHED Rust SDK (`proximadb-sdk = "=X.Y.Z"` from crates.io)
//! against a live server: crate version, health, create, insert, search,
//! delete, drop. Release evidence, part (B).
//!
//! This is NOT a workspace member and never builds against the in-repo crate:
//! release-smoke.yml copies it to a temp dir and writes a Cargo.toml whose only
//! SDK source is the registry (see `render_manifest` in that workflow step).
//!
//! Usage: rust-sdk-smoke <base_url> <expected_version>

use std::time::{Duration, Instant};

use proximadb_sdk::{DistanceMetric, connect};

type BoxErr = Box<dyn std::error::Error>;

fn fail(msg: String) -> BoxErr {
    msg.into()
}

#[tokio::main]
async fn main() {
    if let Err(e) = run().await {
        println!("::error::rust SDK smoke FAILED: {e}");
        std::process::exit(1);
    }
}

async fn search_ids(client: &proximadb_sdk::ProximaClient, name: &str) -> Result<Vec<String>, BoxErr> {
    let hits = client
        .collection(name)
        .search()
        .vector(&[1.0, 0.0, 0.0, 0.0])
        .top_k(3)
        .execute()
        .await?;
    Ok(hits.into_iter().map(|h| h.id).collect())
}

async fn eventually<F, Fut>(what: &str, mut f: F) -> Result<(), BoxErr>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<Result<(), String>, BoxErr>>,
{
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        match f().await? {
            Ok(()) => return Ok(()),
            Err(why) if Instant::now() >= deadline => return Err(fail(format!("{what}: {why}"))),
            Err(_) => tokio::time::sleep(Duration::from_millis(500)).await,
        }
    }
}

async fn run() -> Result<(), BoxErr> {
    let mut args = std::env::args().skip(1);
    let base = args.next().ok_or_else(|| fail("usage: <base_url> <version>".into()))?;
    let want = args.next().ok_or_else(|| fail("usage: <base_url> <version>".into()))?;

    if proximadb_sdk::VERSION != want {
        return Err(fail(format!("proximadb_sdk::VERSION {} != {want}", proximadb_sdk::VERSION)));
    }
    let client = connect(base.as_str())?;

    let health = client.health().await?;
    if health.status != "healthy" || health.version.as_deref() != Some(want.as_str()) {
        return Err(fail(format!("health status={} version={:?}; expected healthy {want}", health.status, health.version)));
    }
    println!("health: ok ({want})");

    let name = format!("rs_rust_{}", std::process::id());
    client.create_collection(&name).dimension(4).metric(DistanceMetric::Cosine).execute().await?;
    println!("create_collection: ok ({name})");

    client
        .collection(&name)
        .insert()
        .batch(
            vec!["a".into(), "b".into(), "c".into()],
            vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0], vec![0.0, 0.0, 1.0, 0.0]],
        )?
        .execute()
        .await?;
    println!("insert: ok");

    eventually("search top-1 should be 'a'", || async {
        let ids = search_ids(&client, &name).await?;
        Ok(if ids.first().map(String::as_str) == Some("a") && ids.len() == 3 { Ok(()) } else { Err(format!("{ids:?}")) })
    })
    .await?;
    println!("search: ok");

    client.collection(&name).delete_vector("a").await?;
    eventually("deleted 'a' must not be returned", || async {
        let ids = search_ids(&client, &name).await?;
        Ok(if ids.iter().any(|i| i == "a") { Err(format!("{ids:?}")) } else { Ok(()) })
    })
    .await?;
    println!("delete + search_after_delete: ok");

    client.delete_collection(&name).await?;
    println!("delete_collection: ok");
    println!("{{\"artifact\":\"crates.io:proximadb-sdk\",\"installed_version\":\"{}\",\"passed\":true}}", proximadb_sdk::VERSION);
    Ok(())
}
