//! Arrow Flight contract gate for the Rust **connectors** that speak Flight
//! (today: Trino + DuckDB-bulk).
//!
//! Unlike the OpenAPI gate (which validates against the YAML spec), the
//! Flight contract is the union of:
//!
//!   - The `arrow-flight` crate's generated `FlightService` proto trait
//!     (pinned via `Cargo.toml`).
//!   - Custom JSON-encoded ticket shapes ProximaDB stamps into
//!     `Ticket.ticket`, e.g. [`ArrowFileTicket`] at
//!     `src/network/arrow_ipc/file_export.rs:639`.
//!   - `FlightDescriptor` path patterns the server's routing logic at
//!     `src/network/arrow_ipc/service.rs:313-342` recognizes
//!     (`["relational", table_fqn]`, `["vectors", collection_id]`, …).
//!
//! Per-method TDD tests for live Flight calls land in C7 (Trino pilot).
//! This file's smoke gate just proves the ticket-shape contract surfaces
//! we'll validate against.

use arrow_flight::{FlightDescriptor, Ticket};
use proximadb::network::arrow_ipc::file_export::ArrowFileTicket;

// ---------------------------------------------------------------------------
// Smoke test — proves the ArrowFileTicket detector recognizes a known-good
// JSON blob. Per-shape contract tests (descriptor patterns, ticket round
// trip) land in C7.
// ---------------------------------------------------------------------------

#[test]
fn helpers_arrow_file_ticket_detector_recognizes_known_shape() {
    let raw = br#"{"type":"arrow_file","collection_id":"c1","file_path":"/tmp/x.arrow"}"#;
    let ticket = Ticket {
        ticket: raw.to_vec().into(),
    };
    assert!(ArrowFileTicket::is_arrow_file_ticket(&ticket));
}

#[test]
fn helpers_arrow_file_ticket_detector_rejects_other_shapes() {
    let raw = br#"{"type":"sql","statement":"SELECT 1"}"#;
    let ticket = Ticket {
        ticket: raw.to_vec().into(),
    };
    assert!(!ArrowFileTicket::is_arrow_file_ticket(&ticket));
}

// ---------------------------------------------------------------------------
// ArrowFileTicket roundtrip — the JSON shape Trino's DoGet codepath needs
// when the server returns split tickets from GetFlightInfo.
// ---------------------------------------------------------------------------

#[test]
fn arrow_file_ticket_json_roundtrips() {
    let original = ArrowFileTicket {
        ticket_type: "arrow_file".to_string(),
        collection_id: "trino_col".to_string(),
        file_path: "/data/segment-001.arrow".to_string(),
        compression: None,
    };
    let bytes = serde_json::to_vec(&original).expect("serialize");
    let ticket = Ticket {
        ticket: bytes.into(),
    };
    assert!(ArrowFileTicket::is_arrow_file_ticket(&ticket));
    let parsed = ArrowFileTicket::from_ticket(&ticket).expect("parse");
    assert_eq!(parsed.collection_id, "trino_col");
    assert_eq!(parsed.file_path, "/data/segment-001.arrow");
}

// ---------------------------------------------------------------------------
// FlightDescriptor path patterns. The server at
// `src/network/arrow_ipc/service.rs:313-342` recognizes two `path` shapes
// when routing a GetSchema / GetFlightInfo request:
//
//   ["relational" | "table" | "sql", <table_fqn>]
//   ["vectors", <collection_id>]
//
// Trino's metadata calls (flight_get_table_schema, flight_get_splits) must
// build descriptors in one of these shapes. The contract gate constructs
// each shape and asserts the proto-level descriptor matches expectation.
// ---------------------------------------------------------------------------

#[test]
fn flight_descriptor_relational_path_shape() {
    let desc = FlightDescriptor::new_path(vec!["relational".into(), "tenant1.users".into()]);
    assert_eq!(
        desc.path,
        vec!["relational".to_string(), "tenant1.users".to_string()]
    );
    // First element must be one of the model-router prefixes the server recognizes.
    let head = desc.path.first().map(String::as_str);
    assert!(matches!(head, Some("relational")));
}

#[test]
fn flight_descriptor_vectors_path_shape() {
    let desc = FlightDescriptor::new_path(vec!["vectors".into(), "embeddings".into()]);
    assert_eq!(
        desc.path,
        vec!["vectors".to_string(), "embeddings".to_string()]
    );
    let head = desc.path.first().map(String::as_str);
    assert!(matches!(head, Some("vectors")));
}

// ---------------------------------------------------------------------------
// TD-098 pilot — in-process tonic Flight server fixture + live call test for
// `flight_list_schemas`. The mock implements `FlightService::list_flights`
// to return canned `FlightInfo` rows; every other RPC returns `Unimplemented`
// (the connector method only uses `list_flights`). Other 6 Trino flight_*
// helpers stay scaffolded; their tests land when the methods become live.
// ---------------------------------------------------------------------------

mod trino_flight_pilot {
    use std::pin::Pin;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use arrow::array::{Int64Array, RecordBatch, StringArray};
    use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
    use arrow_flight::SchemaAsIpc;
    use arrow_flight::flight_service_server::{FlightService, FlightServiceServer};
    use arrow_flight::{
        Action, ActionType, Criteria, Empty, FlightData, FlightDescriptor, FlightInfo,
        HandshakeRequest, HandshakeResponse, PollInfo, PutResult, SchemaResult, Ticket,
    };
    use futures::Stream;
    use proximadb::connectors::trino::{
        TrinoConnectorConfig, TrinoPageSink, TrinoPageSource, TrinoSplit, TrinoTable,
        TrinoTupleDomain, flight_get_splits, flight_get_table_schema, flight_list_schemas,
        flight_list_tables, record_batch_to_trino_page,
    };
    use proximadb::storage::formats::{FileSplit, SplitLocality, SplitStatistics, SplitType};
    use proximadb::storage::schema::ProximaSchema;
    use tokio::sync::oneshot;
    use tokio_stream::wrappers::TcpListenerStream;
    use tonic::{Request, Response, Status, Streaming};

    type RespStream<T> = Pin<Box<dyn Stream<Item = Result<T, Status>> + Send + 'static>>;

    /// Minimal `FlightService` impl. Only `list_flights`, `get_schema`,
    /// `get_flight_info`, `do_get`, and `do_put` are wired; every other
    /// RPC returns Unimplemented to keep the trait satisfied without
    /// 300+ lines of mocks.
    struct MockFlight {
        canned: Vec<FlightInfo>,
        canned_schema: Option<Arc<ArrowSchema>>,
        canned_flight_info: Option<FlightInfo>,
        canned_do_get: Option<Vec<FlightData>>,
        /// When set, `do_put` drains the request stream, increments
        /// this counter per inbound `FlightData`, and emits one
        /// `PutResult` ack per inbound message. When None, `do_put`
        /// returns Unimplemented.
        do_put_count: Option<Arc<AtomicUsize>>,
    }

    #[tonic::async_trait]
    impl FlightService for MockFlight {
        type HandshakeStream = RespStream<HandshakeResponse>;
        type ListFlightsStream = RespStream<FlightInfo>;
        type DoGetStream = RespStream<FlightData>;
        type DoPutStream = RespStream<PutResult>;
        type DoActionStream = RespStream<arrow_flight::Result>;
        type ListActionsStream = RespStream<ActionType>;
        type DoExchangeStream = RespStream<FlightData>;

        async fn handshake(
            &self,
            _request: Request<Streaming<HandshakeRequest>>,
        ) -> Result<Response<Self::HandshakeStream>, Status> {
            Err(Status::unimplemented("handshake"))
        }

        async fn list_flights(
            &self,
            _request: Request<Criteria>,
        ) -> Result<Response<Self::ListFlightsStream>, Status> {
            let items: Vec<Result<FlightInfo, Status>> =
                self.canned.iter().cloned().map(Ok).collect();
            let s = futures::stream::iter(items);
            Ok(Response::new(Box::pin(s)))
        }

        async fn get_flight_info(
            &self,
            _request: Request<FlightDescriptor>,
        ) -> Result<Response<FlightInfo>, Status> {
            let Some(info) = self.canned_flight_info.as_ref() else {
                return Err(Status::unimplemented("get_flight_info"));
            };
            Ok(Response::new(info.clone()))
        }

        async fn poll_flight_info(
            &self,
            _request: Request<FlightDescriptor>,
        ) -> Result<Response<PollInfo>, Status> {
            Err(Status::unimplemented("poll_flight_info"))
        }

        async fn get_schema(
            &self,
            _request: Request<FlightDescriptor>,
        ) -> Result<Response<SchemaResult>, Status> {
            let Some(schema) = self.canned_schema.as_ref() else {
                return Err(Status::unimplemented("get_schema"));
            };
            let options = arrow::ipc::writer::IpcWriteOptions::default();
            let result: SchemaResult = SchemaAsIpc::new(schema.as_ref(), &options)
                .try_into()
                .map_err(|e: arrow::error::ArrowError| Status::internal(e.to_string()))?;
            Ok(Response::new(result))
        }

        async fn do_get(
            &self,
            _request: Request<Ticket>,
        ) -> Result<Response<Self::DoGetStream>, Status> {
            let Some(messages) = self.canned_do_get.as_ref() else {
                return Err(Status::unimplemented("do_get"));
            };
            let items: Vec<Result<FlightData, Status>> = messages.iter().cloned().map(Ok).collect();
            let s = futures::stream::iter(items);
            Ok(Response::new(Box::pin(s)))
        }

        async fn do_put(
            &self,
            request: Request<Streaming<FlightData>>,
        ) -> Result<Response<Self::DoPutStream>, Status> {
            let Some(counter) = self.do_put_count.clone() else {
                return Err(Status::unimplemented("do_put"));
            };
            let mut stream = request.into_inner();
            let (tx, rx) = tokio::sync::mpsc::channel::<Result<PutResult, Status>>(16);
            // Drain the client's FlightData stream and emit one
            // PutResult ack per inbound message. Spawn so the response
            // stream can be returned immediately; the channel closes
            // when the client closes its end (drops the request tx)
            // and the loop exits.
            tokio::spawn(async move {
                use futures::StreamExt;
                while let Some(msg) = stream.next().await {
                    match msg {
                        Ok(_) => {
                            counter.fetch_add(1, Ordering::SeqCst);
                            if tx
                                .send(Ok(PutResult {
                                    app_metadata: Default::default(),
                                }))
                                .await
                                .is_err()
                            {
                                break;
                            }
                        }
                        Err(_) => break,
                    }
                }
            });
            let resp_stream = tokio_stream::wrappers::ReceiverStream::new(rx);
            Ok(Response::new(Box::pin(resp_stream)))
        }

        async fn do_action(
            &self,
            _request: Request<Action>,
        ) -> Result<Response<Self::DoActionStream>, Status> {
            Err(Status::unimplemented("do_action"))
        }

        async fn list_actions(
            &self,
            _request: Request<Empty>,
        ) -> Result<Response<Self::ListActionsStream>, Status> {
            Err(Status::unimplemented("list_actions"))
        }

        async fn do_exchange(
            &self,
            _request: Request<Streaming<FlightData>>,
        ) -> Result<Response<Self::DoExchangeStream>, Status> {
            Err(Status::unimplemented("do_exchange"))
        }
    }

    /// Build a `FlightInfo` whose `flight_descriptor.path` is the given
    /// path segments. Trino's `flight_list_schemas` projects path[0] →
    /// schema name and path[1..] → table name.
    fn flight_info(path: &[&str]) -> FlightInfo {
        FlightInfo {
            schema: Default::default(),
            flight_descriptor: Some(FlightDescriptor::new_path(
                path.iter().map(|s| s.to_string()).collect(),
            )),
            endpoint: vec![],
            total_records: -1,
            total_bytes: -1,
            ordered: false,
            app_metadata: Default::default(),
        }
    }

    /// Spin up a mock FlightService on `127.0.0.1:0` and return the
    /// chosen port + a shutdown signal sender so the test can stop the
    /// server cleanly. The server task self-terminates when the
    /// shutdown receiver fires.
    async fn start_mock_flight_server(canned: Vec<FlightInfo>) -> (u16, oneshot::Sender<()>) {
        start_mock_flight_server_full(canned, None, None, None, None).await
    }

    /// Variant used by tests that need `get_schema` wired (Trino
    /// `flight_get_table_schema` pilot). `canned_schema` is encoded as
    /// Arrow IPC into the `SchemaResult.schema` bytes by the mock.
    async fn start_mock_flight_server_with_schema(
        canned: Vec<FlightInfo>,
        canned_schema: Arc<ArrowSchema>,
    ) -> (u16, oneshot::Sender<()>) {
        start_mock_flight_server_full(canned, Some(canned_schema), None, None, None).await
    }

    /// Variant used by tests that need `get_flight_info` wired (Trino
    /// `flight_get_splits` pilot). The mock returns `canned_flight_info`
    /// verbatim from `get_flight_info`; its endpoints carry the
    /// JSON-encoded splits.
    async fn start_mock_flight_server_with_flight_info(
        canned: Vec<FlightInfo>,
        canned_flight_info: FlightInfo,
    ) -> (u16, oneshot::Sender<()>) {
        start_mock_flight_server_full(canned, None, Some(canned_flight_info), None, None).await
    }

    /// Variant used by tests that need `do_get` wired (Trino
    /// `TrinoPageSource::get_next_page` pilot). The mock streams the
    /// supplied `FlightData` messages back in order (schema message
    /// first, then RecordBatch messages — typically produced via
    /// `arrow_flight::utils::batches_to_flight_data`).
    async fn start_mock_flight_server_with_do_get(
        canned_do_get: Vec<FlightData>,
    ) -> (u16, oneshot::Sender<()>) {
        start_mock_flight_server_full(vec![], None, None, Some(canned_do_get), None).await
    }

    /// Variant used by tests that need `do_put` wired (Trino
    /// `TrinoPageSink::{append_page, finish}` pilot). The mock
    /// increments `do_put_count` per inbound `FlightData` and acks
    /// each with a `PutResult`, so tests can assert how many wire
    /// messages reached the server.
    async fn start_mock_flight_server_with_do_put(
        do_put_count: Arc<AtomicUsize>,
    ) -> (u16, oneshot::Sender<()>) {
        start_mock_flight_server_full(vec![], None, None, None, Some(do_put_count)).await
    }

    async fn start_mock_flight_server_full(
        canned: Vec<FlightInfo>,
        canned_schema: Option<Arc<ArrowSchema>>,
        canned_flight_info: Option<FlightInfo>,
        canned_do_get: Option<Vec<FlightData>>,
        do_put_count: Option<Arc<AtomicUsize>>,
    ) -> (u16, oneshot::Sender<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind port 0");
        let port = listener.local_addr().expect("local_addr").port();
        let incoming = TcpListenerStream::new(listener);

        let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
        let svc = MockFlight {
            canned,
            canned_schema,
            canned_flight_info,
            canned_do_get,
            do_put_count,
        };
        tokio::spawn(async move {
            let _ = tonic::transport::Server::builder()
                .add_service(FlightServiceServer::new(svc))
                .serve_with_incoming_shutdown(incoming, async {
                    let _ = shutdown_rx.await;
                })
                .await;
        });

        // tiny delay so the server is ready when the client dials
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        (port, shutdown_tx)
    }

    #[tokio::test]
    async fn trino_flight_list_schemas_buckets_by_first_path_segment() {
        let canned = vec![
            flight_info(&["sales", "orders"]),
            flight_info(&["sales", "customers"]),
            flight_info(&["analytics", "events"]),
            flight_info(&["analytics"]), // no table — schema-only entry
        ];
        let (port, _shutdown) = start_mock_flight_server(canned).await;

        let config = TrinoConnectorConfig {
            flight_endpoint: format!("grpc://127.0.0.1:{port}"),
            ..TrinoConnectorConfig::default()
        };

        let schemas = flight_list_schemas(&config.flight_endpoint, "proximadb").await;

        // BTreeMap inside flight_list_schemas → schemas come back sorted.
        assert_eq!(schemas.len(), 2, "expected analytics + sales: {schemas:?}");
        let analytics = schemas
            .iter()
            .find(|s| s.name == "analytics")
            .expect("analytics schema");
        assert_eq!(analytics.tables, vec!["events"]);
        let sales = schemas
            .iter()
            .find(|s| s.name == "sales")
            .expect("sales schema");
        // sales has two tables — order is BTreeMap insertion-driven (Vec push)
        assert_eq!(sales.tables.len(), 2);
        assert!(sales.tables.iter().any(|t| t == "orders"));
        assert!(sales.tables.iter().any(|t| t == "customers"));
    }

    #[tokio::test]
    async fn trino_flight_list_schemas_returns_empty_on_unreachable_endpoint() {
        // Dial a port that nothing's listening on — must NOT panic; must
        // degrade to empty schema list.
        let schemas = flight_list_schemas("grpc://127.0.0.1:1", "proximadb").await;
        assert!(schemas.is_empty(), "unreachable endpoint must yield empty");
    }

    #[tokio::test]
    async fn trino_flight_list_tables_filters_by_schema() {
        // Mock returns paths across two schemas + a deduplicate of
        // ["sales","orders"]. flight_list_tables("sales") should pick
        // exactly the two distinct sales tables.
        let canned = vec![
            flight_info(&["sales", "orders"]),
            flight_info(&["analytics", "events"]),
            flight_info(&["sales", "customers"]),
            flight_info(&["sales", "orders"]), // duplicate — must dedup
        ];
        let (port, _shutdown) = start_mock_flight_server(canned).await;

        let endpoint = format!("grpc://127.0.0.1:{port}");
        let tables = flight_list_tables(&endpoint, "proximadb", "sales").await;
        assert_eq!(
            tables.len(),
            2,
            "expected 2 distinct sales tables: {tables:?}"
        );
        assert!(tables.iter().any(|t| t == "orders"));
        assert!(tables.iter().any(|t| t == "customers"));

        // Schema not in canned list → empty Vec.
        let none = flight_list_tables(&endpoint, "proximadb", "missing").await;
        assert!(none.is_empty(), "missing schema must yield empty");
    }

    #[tokio::test]
    async fn trino_flight_list_tables_returns_empty_on_unreachable_endpoint() {
        let tables = flight_list_tables("grpc://127.0.0.1:1", "proximadb", "any").await;
        assert!(tables.is_empty(), "unreachable endpoint must yield empty");
    }

    #[tokio::test]
    async fn trino_flight_get_table_schema_returns_arrow_schema() {
        // Mock get_schema returns a known 2-column Arrow schema; the
        // connector must dial GetSchema and round-trip the IPC bytes
        // back into an ArrowSchema with the same fields.
        let canned_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, true),
        ]));
        let (port, _shutdown) =
            start_mock_flight_server_with_schema(vec![], canned_schema.clone()).await;

        let endpoint = format!("grpc://127.0.0.1:{port}");
        let got = flight_get_table_schema(&endpoint, "proximadb", "sales", "orders")
            .await
            .expect("schema should be returned");
        assert_eq!(got.fields().len(), 2, "expected 2 fields: {got:?}");
        assert_eq!(got.field(0).name(), "id");
        assert_eq!(got.field(0).data_type(), &DataType::Int64);
        assert!(!got.field(0).is_nullable());
        assert_eq!(got.field(1).name(), "name");
        assert_eq!(got.field(1).data_type(), &DataType::Utf8);
        assert!(got.field(1).is_nullable());
    }

    #[tokio::test]
    async fn trino_flight_get_table_schema_returns_none_on_unreachable_endpoint() {
        let result = flight_get_table_schema("grpc://127.0.0.1:1", "proximadb", "any", "any").await;
        assert!(result.is_none(), "unreachable endpoint must yield None");
    }

    /// Build a one-segment FileSplit suitable for round-tripping through
    /// `TrinoSplit::from_file_split` + JSON serde.
    fn sample_file_split(split_id: &str, file_path: &str) -> FileSplit {
        FileSplit {
            split_id: split_id.to_string(),
            file_path: file_path.to_string(),
            offset: 0,
            length: 1024,
            split_type: SplitType::Block {
                block_id: 0,
                record_count: 100,
            },
            statistics: SplitStatistics::default(),
            locality: SplitLocality::default(),
            object_size: None,
        }
    }

    #[tokio::test]
    async fn trino_flight_get_splits_decodes_endpoint_tickets() {
        // Mock returns a FlightInfo with two endpoints; each endpoint's
        // ticket bytes are a JSON-encoded TrinoSplit. The connector
        // must decode each ticket back into a TrinoSplit.
        let split_a = TrinoSplit::from_file_split(
            "proximadb".into(),
            "sales".into(),
            "orders".into(),
            sample_file_split("split-a", "s3://orders/part-0.parquet"),
        );
        let split_b = TrinoSplit::from_file_split(
            "proximadb".into(),
            "sales".into(),
            "orders".into(),
            sample_file_split("split-b", "s3://orders/part-1.parquet"),
        );

        let endpoints = vec![
            arrow_flight::FlightEndpoint {
                ticket: Some(arrow_flight::Ticket {
                    ticket: serde_json::to_vec(&split_a).unwrap().into(),
                }),
                location: vec![],
                expiration_time: None,
                app_metadata: Default::default(),
            },
            arrow_flight::FlightEndpoint {
                ticket: Some(arrow_flight::Ticket {
                    ticket: serde_json::to_vec(&split_b).unwrap().into(),
                }),
                location: vec![],
                expiration_time: None,
                app_metadata: Default::default(),
            },
        ];
        let canned_flight_info = FlightInfo {
            schema: Default::default(),
            flight_descriptor: Some(FlightDescriptor::new_path(vec![
                "sales".into(),
                "orders".into(),
            ])),
            endpoint: endpoints,
            total_records: -1,
            total_bytes: -1,
            ordered: false,
            app_metadata: Default::default(),
        };

        let (port, _shutdown) =
            start_mock_flight_server_with_flight_info(vec![], canned_flight_info).await;

        let endpoint = format!("grpc://127.0.0.1:{port}");
        let splits = flight_get_splits(
            &endpoint,
            "proximadb",
            "sales",
            "orders",
            &TrinoTupleDomain::all(),
        )
        .await;
        assert_eq!(splits.len(), 2, "expected 2 splits: {splits:?}");
        let ids: Vec<&str> = splits.iter().map(|s| s.split_id.as_str()).collect();
        assert!(ids.contains(&"split-a"));
        assert!(ids.contains(&"split-b"));
        let a = splits.iter().find(|s| s.split_id == "split-a").unwrap();
        assert_eq!(a.file_split.file_path, "s3://orders/part-0.parquet");
        assert_eq!(a.schema, "sales");
        assert_eq!(a.table, "orders");
    }

    #[tokio::test]
    async fn trino_flight_get_splits_returns_empty_on_unreachable_endpoint() {
        let splits = flight_get_splits(
            "grpc://127.0.0.1:1",
            "proximadb",
            "any",
            "any",
            &TrinoTupleDomain::all(),
        )
        .await;
        assert!(splits.is_empty(), "unreachable endpoint must yield empty");
    }

    #[tokio::test]
    async fn trino_page_source_streams_batches_from_do_get() {
        // Build 2 canned RecordBatches (4 + 3 rows), encode them as a
        // Flight stream via `batches_to_flight_data`; the connector
        // page source must surface them as 2 sequential TrinoPages
        // and then return None / set `finished=true`.
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, true),
        ]));
        let batch_a = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3, 4])),
                Arc::new(StringArray::from(vec![
                    Some("a"),
                    Some("b"),
                    None,
                    Some("d"),
                ])),
            ],
        )
        .unwrap();
        let batch_b = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(vec![10, 20, 30])),
                Arc::new(StringArray::from(vec!["x", "y", "z"])),
            ],
        )
        .unwrap();
        let messages =
            arrow_flight::utils::batches_to_flight_data(&schema, vec![batch_a, batch_b]).unwrap();

        let (port, _shutdown) = start_mock_flight_server_with_do_get(messages).await;

        let endpoint = format!("grpc://127.0.0.1:{port}");
        let split = TrinoSplit::from_file_split(
            "proximadb".into(),
            "sales".into(),
            "orders".into(),
            sample_file_split("split-a", "s3://orders/part-0.parquet"),
        );

        let mut source = TrinoPageSource::new(&endpoint, split, schema.clone()).await;

        let page1 = source
            .get_next_page()
            .await
            .expect("expected first page from do_get stream");
        assert_eq!(page1.position_count, 4);
        assert_eq!(page1.blocks.len(), 2);
        assert_eq!(page1.blocks[0].column, "id");
        assert_eq!(page1.blocks[1].column, "name");

        let page2 = source
            .get_next_page()
            .await
            .expect("expected second page from do_get stream");
        assert_eq!(page2.position_count, 3);

        // Stream is exhausted; further pulls return None and the
        // source flips to finished.
        assert!(source.get_next_page().await.is_none());
        assert!(source.is_finished());
        assert_eq!(source.get_rows_read(), 7);
        assert!(source.get_bytes_read() > 0, "encoder should have written");
    }

    #[tokio::test]
    async fn trino_page_source_finishes_immediately_on_unreachable_endpoint() {
        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "id",
            DataType::Int64,
            false,
        )]));
        let split = TrinoSplit::from_file_split(
            "proximadb".into(),
            "sales".into(),
            "orders".into(),
            sample_file_split("split-x", "s3://orders/missing.parquet"),
        );
        let mut source = TrinoPageSource::new("grpc://127.0.0.1:1", split, schema).await;
        assert!(
            source.is_finished(),
            "unreachable endpoint must mark finished"
        );
        assert!(source.get_next_page().await.is_none());
    }

    /// Build a `TrinoTable` from an Arrow schema for sink tests.
    fn sample_trino_table(arrow_schema: Arc<ArrowSchema>) -> TrinoTable {
        let proxima_schema = Arc::new(ProximaSchema::from_arrow_schema(
            arrow_schema.as_ref(),
            "test-schema-id".to_string(),
        ));
        TrinoTable {
            schema_name: "sales".to_string(),
            table_name: "orders".to_string(),
            schema: arrow_schema,
            proxima_schema,
            properties: std::collections::HashMap::new(),
            comment: None,
        }
    }

    #[tokio::test]
    async fn trino_page_sink_streams_pages_via_do_put_and_summarizes() {
        // Two TrinoPages (3 + 2 rows). The sink must dial do_put, send
        // a schema FlightData first, then one batch FlightData per
        // page; close on finish; surface a TrinoWriteSummary with the
        // 5-row total. The mock counts every inbound FlightData; we
        // assert it received >=3 messages (1 schema + 2 batches).
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, true),
        ]));
        let table = sample_trino_table(schema.clone());

        let counter = Arc::new(AtomicUsize::new(0));
        let (port, _shutdown) = start_mock_flight_server_with_do_put(counter.clone()).await;

        let endpoint = format!("grpc://127.0.0.1:{port}");
        let mut sink = TrinoPageSink::new(&endpoint, table)
            .await
            .expect("dial do_put");

        let batch_a = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec![Some("a"), None, Some("c")])),
            ],
        )
        .unwrap();
        let batch_b = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(vec![10, 20])),
                Arc::new(StringArray::from(vec!["x", "y"])),
            ],
        )
        .unwrap();
        let page_a = record_batch_to_trino_page(&batch_a).unwrap();
        let page_b = record_batch_to_trino_page(&batch_b).unwrap();

        sink.append_page(page_a).await.expect("append_page a");
        sink.append_page(page_b).await.expect("append_page b");

        let summary = sink.finish().await.expect("finish");
        assert_eq!(summary.rows_written, 5);
        assert!(summary.bytes_written > 0, "should record some bytes");

        // 1 schema + 2 batch messages = 3 FlightData reached the server.
        let received = counter.load(Ordering::SeqCst);
        assert!(
            received >= 3,
            "expected at least 3 FlightData messages, got {received}"
        );
    }

    #[tokio::test]
    async fn trino_page_sink_new_fails_on_unreachable_endpoint() {
        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "id",
            DataType::Int64,
            false,
        )]));
        let table = sample_trino_table(schema);
        let result = TrinoPageSink::new("grpc://127.0.0.1:1", table).await;
        assert!(result.is_err(), "unreachable endpoint must error");
    }

    #[tokio::test]
    async fn trino_page_sink_finish_without_appends_returns_zero_summary() {
        // Dialing succeeds and finish() emits a zero summary when no
        // pages were appended — used by Trino to short-circuit empty
        // partition writers without sending any FlightData.
        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "id",
            DataType::Int64,
            false,
        )]));
        let table = sample_trino_table(schema);
        let counter = Arc::new(AtomicUsize::new(0));
        let (port, _shutdown) = start_mock_flight_server_with_do_put(counter.clone()).await;

        let endpoint = format!("grpc://127.0.0.1:{port}");
        let mut sink = TrinoPageSink::new(&endpoint, table)
            .await
            .expect("dial do_put");
        let summary = sink.finish().await.expect("finish");
        assert_eq!(summary.rows_written, 0);
        assert_eq!(summary.bytes_written, 0);
        assert_eq!(counter.load(Ordering::SeqCst), 0);
    }
}
