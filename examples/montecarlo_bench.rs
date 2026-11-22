//! Monte Carlo option-pricing benchmark — Rust-native I/O + compute vs JVM Spark.
//!
//! Reads option contracts from a Parquet file through ProximaDB's canonical `FileSystem`
//! trait (local `file://` here; `s3://` in production via the same path) and prices them
//! with the `mc_price` DataFusion UDF, comparing:
//!   * a single-thread Rust kernel baseline (no DataFusion, no Arrow),
//!   * a rayon-parallel Rust kernel (the multi-core ceiling),
//!   * a single DataFusion query (CPU-heavy per-row UDF runs in one place → single-threaded),
//!   * Rust-native MPP workers (one DataFusion query per data shard, driven concurrently).
//!
//! It reports options/sec, paths/sec, input MB/sec (I/O), and the I/O-vs-compute split —
//! the evidence for the "comparable to Databricks on a single box" claim. A PySpark
//! reference (`benches/financial/option_pricing_spark.py`) and a numpy baseline run the
//! same workload externally for side-by-side comparison.
//!
//! Run:
//!   cargo run --release --example montecarlo_bench --features datafusion-integration -- \
//!       --contracts 1000000 --paths 100000 --partitions 8 --out /tmp/mc.csv

#[cfg(feature = "datafusion-integration")]
mod imp {
    use std::sync::Arc;
    use std::time::Instant;

    use arrow::array::{ArrayRef, BooleanArray, Float64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use parquet::file::properties::WriterProperties;

    use proximadb::compute::montecarlo::{mc_price_batch, mc_price_european};
    use proximadb::datafusion::{McBenchTiming, benchmark_mc_price_over_parquet};
    use proximadb::storage::persistence::filesystem::FilesystemFactory;

    struct Args {
        contracts: usize,
        paths: usize,
        partitions: usize,
        out: Option<String>,
        data_dir: Option<String>,
    }

    fn parse_args() -> Args {
        let mut a = Args {
            contracts: 100_000,
            paths: 10_000,
            partitions: num_cpus::get(),
            out: None,
            data_dir: None,
        };
        let argv: Vec<String> = std::env::args().collect();
        let mut i = 1;
        while i < argv.len() {
            let val = |i: usize| argv.get(i + 1).cloned().unwrap_or_default();
            match argv[i].as_str() {
                "--contracts" => {
                    a.contracts = val(i).parse().unwrap_or(a.contracts);
                    i += 2;
                }
                "--paths" => {
                    a.paths = val(i).parse().unwrap_or(a.paths);
                    i += 2;
                }
                "--partitions" => {
                    a.partitions = val(i).parse().unwrap_or(a.partitions);
                    i += 2;
                }
                "--out" => {
                    a.out = Some(val(i));
                    i += 2;
                }
                "--data-dir" => {
                    a.data_dir = Some(val(i));
                    i += 2;
                }
                _ => i += 1,
            }
        }
        a
    }

    struct Contract {
        id: String,
        spot: f64,
        strike: f64,
        vol: f64,
        rate: f64,
        t: f64,
        is_call: bool,
    }

    fn make_contracts(n: usize) -> Vec<Contract> {
        let strikes = [90.0, 95.0, 100.0, 105.0, 110.0];
        let vols = [0.15, 0.2, 0.25, 0.3];
        let ts = [0.25, 0.5, 1.0, 2.0];
        (0..n)
            .map(|i| Contract {
                id: format!("opt_{i:08}"),
                spot: 100.0,
                strike: strikes[i % strikes.len()],
                vol: vols[(i / strikes.len()) % vols.len()],
                rate: 0.03,
                t: ts[(i / (strikes.len() * vols.len())) % ts.len()],
                is_call: i % 2 == 0,
            })
            .collect()
    }

    fn write_parquet(path: &std::path::Path, contracts: &[Contract]) -> u64 {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("spot", DataType::Float64, false),
            Field::new("strike", DataType::Float64, false),
            Field::new("vol", DataType::Float64, false),
            Field::new("rate", DataType::Float64, false),
            Field::new("t", DataType::Float64, false),
            Field::new("is_call", DataType::Boolean, false),
        ]));
        let columns: Vec<ArrayRef> = vec![
            Arc::new(StringArray::from(
                contracts.iter().map(|c| c.id.clone()).collect::<Vec<_>>(),
            )),
            Arc::new(Float64Array::from(
                contracts.iter().map(|c| c.spot).collect::<Vec<_>>(),
            )),
            Arc::new(Float64Array::from(
                contracts.iter().map(|c| c.strike).collect::<Vec<_>>(),
            )),
            Arc::new(Float64Array::from(
                contracts.iter().map(|c| c.vol).collect::<Vec<_>>(),
            )),
            Arc::new(Float64Array::from(
                contracts.iter().map(|c| c.rate).collect::<Vec<_>>(),
            )),
            Arc::new(Float64Array::from(
                contracts.iter().map(|c| c.t).collect::<Vec<_>>(),
            )),
            Arc::new(BooleanArray::from(
                contracts.iter().map(|c| c.is_call).collect::<Vec<_>>(),
            )),
        ];
        let batch = RecordBatch::try_new(schema.clone(), columns).unwrap();
        let props = WriterProperties::builder()
            .set_max_row_group_row_count(Some(8192))
            .build();
        let file = std::fs::File::create(path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
        std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
    }

    /// Single-thread Rust kernel baseline: sequential pricing, no DataFusion, no Arrow.
    fn baseline_single_thread(
        contracts: &[Contract],
        n_paths: usize,
    ) -> (f64, std::time::Duration) {
        let start = Instant::now();
        let mut acc = 0.0f64;
        for (i, c) in contracts.iter().enumerate() {
            acc += mc_price_european(
                c.spot,
                c.strike,
                c.vol,
                c.rate,
                c.t,
                c.is_call,
                n_paths,
                i as u64 + 1,
            );
        }
        // `acc` is returned to prevent the loop from being optimized away.
        (acc, start.elapsed())
    }

    /// Rayon-parallel Rust kernel baseline (all cores, no DataFusion, no Arrow) — the raw
    /// compute ceiling the engine paths are measured against.
    fn baseline_parallel(contracts: &[Contract], n_paths: usize) -> (f64, std::time::Duration) {
        let spot: Vec<f64> = contracts.iter().map(|c| c.spot).collect();
        let strike: Vec<f64> = contracts.iter().map(|c| c.strike).collect();
        let vol: Vec<f64> = contracts.iter().map(|c| c.vol).collect();
        let rate: Vec<f64> = contracts.iter().map(|c| c.rate).collect();
        let t: Vec<f64> = contracts.iter().map(|c| c.t).collect();
        let is_call: Vec<bool> = contracts.iter().map(|c| c.is_call).collect();
        let start = Instant::now();
        let prices = mc_price_batch(&spot, &strike, &vol, &rate, &t, &is_call, n_paths, 1);
        let dur = start.elapsed();
        (prices.iter().sum(), dur)
    }

    fn report(
        mode: &str,
        contracts: usize,
        paths: usize,
        partitions: usize,
        wall_secs: f64,
        input_mb: f64,
        io_secs: Option<f64>,
        out: &mut Option<std::fs::File>,
    ) {
        use std::io::Write;
        let opts_per_sec = contracts as f64 / wall_secs;
        let paths_per_sec = (contracts as f64 * paths as f64) / wall_secs;
        let io_mbps = io_secs.map(|s| input_mb / s).unwrap_or(0.0);
        println!(
            "{mode:<22} contracts={contracts} paths={paths} partitions={partitions} \
             wall={wall_secs:.3}s  options/s={opts_per_sec:>12.0}  paths/s={paths_per_sec:>15.0}  \
             io={io_mbps:>7.1} MB/s",
        );
        if let Some(f) = out.as_mut() {
            let _ = writeln!(
                f,
                "{mode},{contracts},{paths},{partitions},{wall_secs:.4},{opts_per_sec:.1},{paths_per_sec:.1},{input_mb:.3},{io_mbps:.2}",
            );
        }
    }

    pub fn run() -> anyhow::Result<()> {
        let args = parse_args();
        println!(
            "Monte Carlo option-pricing benchmark — cores={}, contracts={}, paths={}, partitions={}",
            num_cpus::get(),
            args.contracts,
            args.paths,
            args.partitions
        );

        let tmp = tempfile::tempdir()?;
        let dir = args
            .data_dir
            .clone()
            .unwrap_or_else(|| tmp.path().display().to_string());
        std::fs::create_dir_all(&dir)?;
        let parquet_path = std::path::Path::new(&dir).join("options.parquet");

        let contracts = make_contracts(args.contracts);
        let file_size = write_parquet(&parquet_path, &contracts);
        let input_mb = file_size as f64 / 1.0e6;
        println!(
            "Wrote {} contracts to {} ({:.2} MB)\n",
            args.contracts,
            parquet_path.display(),
            input_mb
        );

        let mut out_file = match &args.out {
            Some(p) => {
                use std::io::Write;
                let mut f = std::fs::File::create(p)?;
                writeln!(
                    f,
                    "mode,contracts,paths,partitions,wall_secs,options_per_sec,paths_per_sec,input_mb,io_mbps"
                )?;
                Some(f)
            }
            None => None,
        };

        // 1) Single-thread Rust kernel (no DataFusion / no Arrow).
        let (sink, base_dur) = baseline_single_thread(&contracts, args.paths);
        std::hint::black_box(sink);
        report(
            "kernel-1thread",
            args.contracts,
            args.paths,
            1,
            base_dur.as_secs_f64(),
            input_mb,
            None,
            &mut out_file,
        );

        // 2) Rayon-parallel Rust kernel (all cores, no DataFusion / no Arrow).
        let (sink, par_dur) = baseline_parallel(&contracts, args.paths);
        std::hint::black_box(sink);
        report(
            "kernel-parallel",
            args.contracts,
            args.paths,
            num_cpus::get(),
            par_dur.as_secs_f64(),
            input_mb,
            None,
            &mut out_file,
        );

        // DataFusion over Parquet read through the FileSystem trait, two ways:
        //   3) single-query DataFusion (one SessionContext) — a CPU-heavy per-row UDF runs in
        //      one place in the query's stream graph, so it is effectively single-threaded;
        //   4) Rust-native MPP workers — the contracts are sharded into N Parquet files and a
        //      coordinator drives one independent DataFusion query per shard concurrently.
        //      This is the model the data-warehouse course-correction prescribes for
        //      CPU-intensive ("Monte Carlo style") jobs, and it is what actually scales.
        let url = format!("file://{}", parquet_path.display());
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(async {
            let factory = FilesystemFactory::create_default()
                .await
                .map_err(|e| anyhow::anyhow!("filesystem factory: {e}"))?;
            let fs = factory
                .get_filesystem(&url)
                .map_err(|e| anyhow::anyhow!("get_filesystem: {e}"))?;

            // 3) Single-query DataFusion baseline.
            let McBenchTiming {
                rows,
                compute,
                register,
                ..
            } = benchmark_mc_price_over_parquet(fs.clone(), &url, args.paths, 1)
                .await
                .map_err(|e| anyhow::anyhow!("benchmark: {e}"))?;
            assert_eq!(rows, args.contracts, "every contract must be priced");
            report(
                "datafusion-1query",
                args.contracts,
                args.paths,
                1,
                compute.as_secs_f64(),
                input_mb,
                Some(register.as_secs_f64()),
                &mut out_file,
            );

            // 4) Rust-native MPP workers: shard contracts, one DataFusion query per shard.
            let n_workers = args.partitions.max(1);
            let mut shard_urls = Vec::with_capacity(n_workers);
            let shard_size = args.contracts.div_ceil(n_workers);
            for (w, chunk) in contracts.chunks(shard_size).enumerate() {
                let p = std::path::Path::new(&dir).join(format!("options_shard_{w}.parquet"));
                write_parquet(&p, chunk);
                shard_urls.push(format!("file://{}", p.display()));
            }

            let start = Instant::now();
            let mut handles = Vec::with_capacity(shard_urls.len());
            for surl in shard_urls {
                let fs = fs.clone();
                let paths = args.paths;
                handles.push(tokio::spawn(async move {
                    benchmark_mc_price_over_parquet(fs, &surl, paths, 1).await
                }));
            }
            let mut total_rows = 0usize;
            for h in handles {
                let timing = h
                    .await
                    .map_err(|e| anyhow::anyhow!("worker join: {e}"))?
                    .map_err(|e| anyhow::anyhow!("worker query: {e}"))?;
                total_rows += timing.rows;
            }
            let workers_wall = start.elapsed();
            assert_eq!(total_rows, args.contracts, "all shards must be priced");
            report(
                &format!("datafusion-{n_workers}workers"),
                args.contracts,
                args.paths,
                n_workers,
                workers_wall.as_secs_f64(),
                input_mb,
                None,
                &mut out_file,
            );

            Ok::<(), anyhow::Error>(())
        })?;

        if let Some(p) = &args.out {
            println!("\nCSV written to {p}");
        }
        Ok(())
    }
}

fn main() {
    #[cfg(feature = "datafusion-integration")]
    {
        if let Err(e) = imp::run() {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    }
    #[cfg(not(feature = "datafusion-integration"))]
    {
        eprintln!(
            "This benchmark requires the datafusion-integration feature:\n  \
             cargo run --release --example montecarlo_bench --features datafusion-integration -- --contracts 100000 --paths 10000"
        );
    }
}
