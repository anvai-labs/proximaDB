//! # Python DataFrame API for ProximaDB (Proxima-Spark)
//!
//! Provides a Rust-native distributed execution engine based on DataFusion
//! with a Python front-end via PyO3.

use crate::EmbeddedProximaDB;
use arrow::pyarrow::ToPyArrow;
use datafusion::logical_expr::{Expr, ExprSchemable};
use datafusion::prelude::*;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::sync::Arc;

fn require_non_blank(label: &str, value: &str) -> PyResult<()> {
    if value.trim().is_empty() {
        Err(PyValueError::new_err(format!(
            "{} must not be empty",
            label
        )))
    } else {
        Ok(())
    }
}

#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct PyExpr {
    pub expr: Expr,
    sort_options: Option<(bool, bool)>,
}

#[pymethods]
impl PyExpr {
    #[new]
    fn new(expr_str: String) -> PyResult<Self> {
        require_non_blank("Column name", &expr_str)?;
        Ok(Self {
            expr: col(expr_str),
            sort_options: None,
        })
    }

    /// Give this expression an alias
    fn alias(&self, name: String) -> PyResult<Self> {
        require_non_blank("Alias name", &name)?;
        Ok(Self {
            expr: self.expr.clone().alias(name),
            sort_options: None,
        })
    }

    /// Cast this expression to a different data type
    fn cast(&self, to_type: String) -> PyResult<Self> {
        let dt = match to_type.to_lowercase().as_str() {
            "int" | "int64" | "bigint" => datafusion::arrow::datatypes::DataType::Int64,
            "int32" => datafusion::arrow::datatypes::DataType::Int32,
            "float" | "float64" | "double" => datafusion::arrow::datatypes::DataType::Float64,
            "float32" => datafusion::arrow::datatypes::DataType::Float32,
            "string" | "utf8" | "text" => datafusion::arrow::datatypes::DataType::Utf8,
            "bool" | "boolean" => datafusion::arrow::datatypes::DataType::Boolean,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unsupported cast type: {}",
                    to_type
                )));
            }
        };
        let empty_schema = datafusion::common::DFSchema::empty();
        Ok(Self {
            expr: self
                .expr
                .clone()
                .cast_to(&dt, &empty_schema)
                .map_err(|e| PyRuntimeError::new_err(format!("{}", e)))?,
            sort_options: None,
        })
    }

    /// Check if this expression is null
    fn is_null(&self) -> Self {
        Self {
            expr: self.expr.clone().is_null(),
            sort_options: None,
        }
    }

    /// Build a sort expression for DataFrame.sort().
    #[pyo3(signature = (ascending=true, nulls_first=false))]
    fn sort(&self, ascending: bool, nulls_first: bool) -> Self {
        Self {
            expr: self.expr.clone(),
            sort_options: Some((ascending, nulls_first)),
        }
    }

    // Operator Overloading

    fn __add__(&self, rhs: PyExpr) -> Self {
        Self {
            expr: self.expr.clone() + rhs.expr,
            sort_options: None,
        }
    }

    fn __sub__(&self, rhs: PyExpr) -> Self {
        Self {
            expr: self.expr.clone() - rhs.expr,
            sort_options: None,
        }
    }

    fn __mul__(&self, rhs: PyExpr) -> Self {
        Self {
            expr: self.expr.clone() * rhs.expr,
            sort_options: None,
        }
    }

    fn __truediv__(&self, rhs: PyExpr) -> Self {
        Self {
            expr: self.expr.clone() / rhs.expr,
            sort_options: None,
        }
    }

    fn __eq__(&self, rhs: PyExpr) -> Self {
        Self {
            expr: self.expr.clone().eq(rhs.expr),
            sort_options: None,
        }
    }

    fn __ne__(&self, rhs: PyExpr) -> Self {
        Self {
            expr: self.expr.clone().not_eq(rhs.expr),
            sort_options: None,
        }
    }

    fn __gt__(&self, rhs: PyExpr) -> Self {
        Self {
            expr: self.expr.clone().gt(rhs.expr),
            sort_options: None,
        }
    }

    fn __ge__(&self, rhs: PyExpr) -> Self {
        Self {
            expr: self.expr.clone().gt_eq(rhs.expr),
            sort_options: None,
        }
    }

    fn __lt__(&self, rhs: PyExpr) -> Self {
        Self {
            expr: self.expr.clone().lt(rhs.expr),
            sort_options: None,
        }
    }

    fn __le__(&self, rhs: PyExpr) -> Self {
        Self {
            expr: self.expr.clone().lt_eq(rhs.expr),
            sort_options: None,
        }
    }

    fn __and__(&self, rhs: PyExpr) -> Self {
        Self {
            expr: self.expr.clone().and(rhs.expr),
            sort_options: None,
        }
    }

    fn __or__(&self, rhs: PyExpr) -> Self {
        Self {
            expr: self.expr.clone().or(rhs.expr),
            sort_options: None,
        }
    }

    fn __invert__(&self) -> Self {
        Self {
            expr: self.expr.clone().not(),
            sort_options: None,
        }
    }

    fn __repr__(&self) -> String {
        format!("PyExpr({})", self.expr)
    }
}

/// Create a column expression
#[pyfunction]
pub fn py_col(name: String) -> PyResult<PyExpr> {
    require_non_blank("Column name", &name)?;
    Ok(PyExpr {
        expr: col(name),
        sort_options: None,
    })
}

/// Create a literal expression
#[pyfunction]
pub fn py_lit(value: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
    if value.is_none() {
        Ok(PyExpr {
            expr: Expr::Literal(datafusion::common::ScalarValue::Null, None),
            sort_options: None,
        })
    } else if let Ok(b) = value.extract::<bool>() {
        Ok(PyExpr {
            expr: lit(b),
            sort_options: None,
        })
    } else if let Ok(i) = value.extract::<i64>() {
        Ok(PyExpr {
            expr: lit(i),
            sort_options: None,
        })
    } else if let Ok(f) = value.extract::<f64>() {
        Ok(PyExpr {
            expr: lit(f),
            sort_options: None,
        })
    } else if let Ok(s) = value.extract::<String>() {
        Ok(PyExpr {
            expr: lit(s),
            sort_options: None,
        })
    } else {
        Err(PyValueError::new_err("Unsupported literal type"))
    }
}

// Aggregate Functions (using datafusion::functions_aggregate)

#[pyfunction]
pub fn py_count(expr: PyExpr) -> PyExpr {
    PyExpr {
        expr: datafusion::functions_aggregate::expr_fn::count(expr.expr),
        sort_options: None,
    }
}

#[pyfunction]
pub fn py_sum(expr: PyExpr) -> PyExpr {
    PyExpr {
        expr: datafusion::functions_aggregate::expr_fn::sum(expr.expr),
        sort_options: None,
    }
}

#[pyfunction]
pub fn py_avg(expr: PyExpr) -> PyExpr {
    PyExpr {
        expr: datafusion::functions_aggregate::expr_fn::avg(expr.expr),
        sort_options: None,
    }
}

#[pyfunction]
pub fn py_min(expr: PyExpr) -> PyExpr {
    PyExpr {
        expr: datafusion::functions_aggregate::expr_fn::min(expr.expr),
        sort_options: None,
    }
}

#[pyfunction]
pub fn py_max(expr: PyExpr) -> PyExpr {
    PyExpr {
        expr: datafusion::functions_aggregate::expr_fn::max(expr.expr),
        sort_options: None,
    }
}

#[pyclass]
pub struct PyDataFusionSession {
    ctx: SessionContext,
    db: Arc<EmbeddedProximaDB>,
}

impl PyDataFusionSession {
    pub fn new(ctx: SessionContext, db: Arc<EmbeddedProximaDB>) -> Self {
        Self { ctx, db }
    }

    fn collection_arrow_schema(
        &self,
        table_name: &str,
        dimension: u32,
    ) -> PyResult<arrow::datatypes::SchemaRef> {
        use proximadb::datafusion::infer_schema_from_collection;
        use proximadb_storage_common::proxima_schema::ProximaSchema;

        let proxima_schema = self.db.runtime().block_on(async {
            match self
                .db
                .shared_services()
                .catalog_manager
                .resolve_table(table_name)
                .await
            {
                Ok((catalog, id)) => catalog.get_table(&id).await.ok(),
                Err(_) => None,
            }
        });

        let schema =
            proxima_schema.unwrap_or_else(|| ProximaSchema::vector_record_schema(dimension));
        infer_schema_from_collection(&schema).map_err(|e| {
            PyRuntimeError::new_err(format!(
                "Schema inference failed for table '{}': {}",
                table_name, e
            ))
        })
    }
}

#[pymethods]
impl PyDataFusionSession {
    /// Execute a SQL query and return a DataFrame
    fn sql(&self, _py: Python<'_>, query: String) -> PyResult<PyDataFrame> {
        require_non_blank("SQL query", &query)?;
        let df = self
            .db
            .runtime()
            .block_on(async { self.ctx.sql(&query).await })
            .map_err(|e| PyRuntimeError::new_err(format!("DataFusion error: {}", e)))?;

        Ok(PyDataFrame {
            df: Arc::new(df),
            db: self.db.clone(),
        })
    }

    /// Register a ProximaDB collection as a table in the DataFusion session
    fn table(&self, _py: Python<'_>, name: String) -> PyResult<PyDataFrame> {
        require_non_blank("Table name", &name)?;
        let df = self
            .db
            .runtime()
            .block_on(async { self.ctx.table(&name).await })
            .map_err(|e| PyRuntimeError::new_err(format!("DataFusion error: {}", e)))?;

        Ok(PyDataFrame {
            df: Arc::new(df),
            db: self.db.clone(),
        })
    }

    /// Register all ProximaDB collections as tables in the DataFusion session
    fn refresh_tables(&self, _py: Python<'_>) -> PyResult<()> {
        use proximadb::datafusion::CollectionInfo;
        use proximadb::datafusion::ProximaDataFusionTable;
        use std::sync::Arc;

        let collections = self
            .db
            .list_collections()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to list collections: {}", e)))?;

        for info in collections {
            let table_name = info.name.clone();

            // 1. Get the catalog-authoritative schema when available. Plain
            // vector collections created through the embedded API may not have
            // a relational table schema yet, so fall back to the canonical
            // vector-record shape instead of registering an empty schema.
            let schema = self.collection_arrow_schema(&table_name, info.dimension)?;

            // 2. Create collection info for DataFusion
            let df_info = CollectionInfo::new(
                table_name.clone(),
                info.dimension as usize,
                match info.engine.as_str() {
                    "helix" => proximadb::datafusion::EngineType::Helix,
                    "nova" => proximadb::datafusion::EngineType::Nova,
                    _ => proximadb::datafusion::EngineType::Sst,
                },
            );

            // 3. Create appropriate table provider for the engine
            let filesystem_factory = self.db.shared_services().filesystem_factory.clone();

            // For embedded mode, we use the local data directory
            let base_path = format!(
                "file://{}/data/{}",
                self.db.config.storage_locations[0]
                    .path
                    .trim_end_matches('/'),
                table_name
            );

            let table: Arc<dyn datafusion::datasource::TableProvider> = match df_info.engine_type {
                proximadb::datafusion::EngineType::Sst => Arc::new(
                    proximadb::datafusion::engine_adapters::SstTableProvider::new(
                        df_info.clone(),
                        base_path.clone(),
                        filesystem_factory.clone(),
                    ),
                ),
                proximadb::datafusion::EngineType::Helix => Arc::new(
                    proximadb::datafusion::engine_adapters::HelixTableProvider::new(
                        df_info.clone(),
                        base_path.clone(),
                        filesystem_factory.clone(),
                    ),
                ),
                _ => {
                    let reader = Arc::new(
                        proximadb::datafusion::proxima_scan_exec::NullSplitReader::new(
                            schema.clone(),
                            df_info.engine_type.clone(),
                        ),
                    );
                    Arc::new(ProximaDataFusionTable::new(
                        table_name.clone(),
                        df_info,
                        schema,
                        reader,
                    ))
                }
            };

            // 4. Create and register table. Refreshing is expected to be safe
            // in long-lived notebook sessions, so replace an existing provider
            // for the collection instead of treating duplicate registration as
            // fatal.
            self.ctx.deregister_table(&table_name).map_err(|e| {
                PyRuntimeError::new_err(format!(
                    "Failed to refresh table {} before registration: {}",
                    table_name, e
                ))
            })?;
            self.ctx.register_table(&table_name, table).map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to register table {}: {}", table_name, e))
            })?;
        }

        Ok(())
    }

    /// Perform a vector similarity search natively in DataFusion.
    /// This utilizes the `vector_search` UDTF to return a DataFrame of `(id, score)`
    /// that can be joined with other relational data.
    #[pyo3(signature = (collection, query_vector, k=10))]
    fn vector_search(
        &self,
        py: Python<'_>,
        collection: String,
        query_vector: Vec<f32>,
        k: u32,
    ) -> PyResult<PyDataFrame> {
        require_non_blank("Collection name", &collection)?;
        if query_vector.is_empty() {
            return Err(PyValueError::new_err(
                "Query vector must contain at least one dimension",
            ));
        }
        if k == 0 {
            return Err(PyValueError::new_err("k must be greater than zero"));
        }

        let escaped_collection = collection.replace('\'', "''");
        let vector_str = query_vector
            .iter()
            .map(|value| value.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        let vector_literal = format!("[{}]", vector_str);

        let sql = format!(
            "SELECT * FROM vector_search('{}', '{}', {})",
            escaped_collection, vector_literal, k
        );

        self.sql(py, sql)
    }

    /// Execute a query across the entire ProximaDB cluster (Distributed)
    fn execute_distributed(&self, py: Python<'_>, query: String) -> PyResult<PyDataFrame> {
        self.sql(py, query)
    }
}

#[pyclass]
pub struct PyDataFrame {
    df: Arc<DataFrame>,
    db: Arc<EmbeddedProximaDB>,
}

#[pymethods]
impl PyDataFrame {
    /// Select columns from the DataFrame
    #[pyo3(signature = (*exprs))]
    fn select(&self, exprs: Vec<PyExpr>) -> PyResult<Self> {
        let df_exprs = exprs.into_iter().map(|e| e.expr).collect::<Vec<_>>();
        let df = (*self.df).clone();
        let new_df = df
            .select(df_exprs)
            .map_err(|e| PyRuntimeError::new_err(format!("DataFusion error: {}", e)))?;

        Ok(Self {
            df: Arc::new(new_df),
            db: self.db.clone(),
        })
    }

    /// Filter the DataFrame
    fn filter(&self, predicate: PyExpr) -> PyResult<Self> {
        let df = (*self.df).clone();
        let new_df = df
            .filter(predicate.expr)
            .map_err(|e| PyRuntimeError::new_err(format!("DataFusion error: {}", e)))?;

        Ok(Self {
            df: Arc::new(new_df),
            db: self.db.clone(),
        })
    }

    /// Alias for filter
    #[pyo3(name = "where")]
    fn py_where(&self, predicate: PyExpr) -> PyResult<Self> {
        self.filter(predicate)
    }

    /// Limit the number of rows
    fn limit(&self, n: usize) -> PyResult<Self> {
        let df = (*self.df).clone();
        let new_df = df
            .limit(0, Some(n))
            .map_err(|e| PyRuntimeError::new_err(format!("DataFusion error: {}", e)))?;

        Ok(Self {
            df: Arc::new(new_df),
            db: self.db.clone(),
        })
    }

    /// Sort the DataFrame
    #[pyo3(signature = (*exprs))]
    fn sort(&self, exprs: Vec<PyExpr>) -> PyResult<Self> {
        let sort_exprs = exprs
            .into_iter()
            .map(|e| {
                let (ascending, nulls_first) = e.sort_options.unwrap_or((true, false));
                e.expr.sort(ascending, nulls_first)
            })
            .collect::<Vec<_>>();
        let df = (*self.df).clone();

        let new_df = df
            .sort(sort_exprs)
            .map_err(|e| PyRuntimeError::new_err(format!("DataFusion error: {}", e)))?;

        Ok(Self {
            df: Arc::new(new_df),
            db: self.db.clone(),
        })
    }

    /// Aggregate the DataFrame
    fn aggregate(&self, group_exprs: Vec<PyExpr>, agg_exprs: Vec<PyExpr>) -> PyResult<Self> {
        if agg_exprs.is_empty() {
            return Err(PyValueError::new_err(
                "aggregate requires at least one aggregate expression",
            ));
        }
        let df_group = group_exprs.into_iter().map(|e| e.expr).collect::<Vec<_>>();
        let df_agg = agg_exprs.into_iter().map(|e| e.expr).collect::<Vec<_>>();
        let df = (*self.df).clone();

        let new_df = df
            .aggregate(df_group, df_agg)
            .map_err(|e| PyRuntimeError::new_err(format!("DataFusion error: {}", e)))?;

        Ok(Self {
            df: Arc::new(new_df),
            db: self.db.clone(),
        })
    }

    /// Join with another DataFrame
    #[pyo3(signature = (other, on, how="inner"))]
    fn join(&self, other: &PyDataFrame, on: Vec<String>, how: &str) -> PyResult<Self> {
        if on.is_empty() {
            return Err(PyValueError::new_err("join requires at least one key"));
        }
        for key in &on {
            require_non_blank("Join key", key)?;
        }

        let join_type = match how.trim().to_lowercase().as_str() {
            "inner" => datafusion::logical_expr::JoinType::Inner,
            "left" => datafusion::logical_expr::JoinType::Left,
            "right" => datafusion::logical_expr::JoinType::Right,
            "full" => datafusion::logical_expr::JoinType::Full,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unsupported join type: {}",
                    how
                )));
            }
        };

        let left_cols = on.iter().map(|s| s.as_str()).collect::<Vec<_>>();
        let right_cols = on.iter().map(|s| s.as_str()).collect::<Vec<_>>();
        let df = (*self.df).clone();

        let new_df = df
            .join(
                other.df.as_ref().clone(),
                join_type,
                &left_cols,
                &right_cols,
                None,
            )
            .map_err(|e| PyRuntimeError::new_err(format!("DataFusion error: {}", e)))?;

        Ok(Self {
            df: Arc::new(new_df),
            db: self.db.clone(),
        })
    }

    /// Add a new column or replace an existing one
    fn with_column(&self, name: String, expr: PyExpr) -> PyResult<Self> {
        require_non_blank("Column name", &name)?;
        let df = (*self.df).clone();
        let new_df = df
            .with_column(&name, expr.expr)
            .map_err(|e| PyRuntimeError::new_err(format!("DataFusion error: {}", e)))?;

        Ok(Self {
            df: Arc::new(new_df),
            db: self.db.clone(),
        })
    }

    /// Convert results to a PyArrow Table (Zero-Copy)
    fn to_arrow(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let batches = self
            .db
            .runtime()
            .block_on(async { (*self.df).clone().collect().await })
            .map_err(|e| PyRuntimeError::new_err(format!("DataFusion error: {}", e)))?;

        let pyarrow = py.import("pyarrow")?;
        let py_batches = PyList::empty(py);

        for batch in batches {
            let py_batch = batch.to_pyarrow(py)?;
            py_batches.append(py_batch)?;
        }

        let schema = self.df.schema().as_arrow().to_pyarrow(py)?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("schema", schema)?;
        let table =
            pyarrow
                .getattr("Table")?
                .call_method("from_batches", (py_batches,), Some(&kwargs))?;
        Ok(table.into_any().unbind())
    }

    /// Collect results as a list of dictionaries
    fn collect(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let batches = self
            .db
            .runtime()
            .block_on(async { (*self.df).clone().collect().await })
            .map_err(|e| PyRuntimeError::new_err(format!("DataFusion error: {}", e)))?;

        let list = PyList::empty(py);
        for batch in batches {
            let py_batch = batch.to_pyarrow(py)?;
            let pylist = py_batch.call_method0("to_pylist")?;
            for item in pylist.cast::<PyList>()?.iter() {
                list.append(item)?;
            }
        }

        Ok(list.into_any().unbind())
    }

    /// Show the first N rows of the DataFrame
    fn show(&self, n: usize) -> PyResult<()> {
        self.db
            .runtime()
            .block_on(async {
                let df = (*self.df).clone();
                df.limit(0, Some(n))
                    .map_err(|e| datafusion::error::DataFusionError::from(e))?
                    .show()
                    .await
            })
            .map_err(|e| PyRuntimeError::new_err(format!("DataFusion error: {}", e)))?;
        Ok(())
    }
}

pub fn register_dataframe_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDataFusionSession>()?;
    m.add_class::<PyDataFrame>()?;
    m.add_class::<PyExpr>()?;
    m.add_function(wrap_pyfunction!(py_col, m)?)?;
    m.add_function(wrap_pyfunction!(py_lit, m)?)?;
    m.add_function(wrap_pyfunction!(py_count, m)?)?;
    m.add_function(wrap_pyfunction!(py_sum, m)?)?;
    m.add_function(wrap_pyfunction!(py_avg, m)?)?;
    m.add_function(wrap_pyfunction!(py_min, m)?)?;
    m.add_function(wrap_pyfunction!(py_max, m)?)?;
    Ok(())
}
