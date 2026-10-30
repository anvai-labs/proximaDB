//! Generic implementations of base traits for common use cases

use super::base_traits::*;
use chrono::{DateTime, Utc};
use serde::de::{self, MapAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::HashMap;
use std::fmt;
use std::fmt::Debug;
use std::marker::PhantomData;

/// Serialize a fixed list of named fields as a struct — shared body of the
/// manual `Serialize` impls below (struct name, field order, and count are
/// preserved verbatim by each caller).
macro_rules! serialize_fields {
    ($serializer:expr, $name:expr, $( $field:expr => $value:expr ),+ $(,)?) => {{
        use serde::ser::SerializeStruct;
        let mut state = $serializer.serialize_struct($name, [$( $field ),+].len())?;
        $( state.serialize_field($field, &$value)?; )+
        state.end()
    }};
}

/// Read the next map value into `slot`, rejecting a repeated field — the
/// shared shape of every field arm in the manual `Deserialize` visitors below.
macro_rules! take_field {
    ($map:expr, $slot:expr, $name:expr) => {{
        if $slot.is_some() {
            return Err(de::Error::duplicate_field($name));
        }
        $slot = Some($map.next_value()?);
    }};
}

/// Generic configuration wrapper that implements BaseConfig
#[derive(Debug, Clone)]
pub struct GenericConfig<T> {
    /// The wrapped configuration data
    pub data: T,
    /// Mapping of validation rule names to error messages
    pub validation_rules: HashMap<String, String>,
    _phantom: PhantomData<T>,
}

impl<T> GenericConfig<T>
where
    T: Debug + Clone + Serialize + for<'de> Deserialize<'de> + Send + Sync,
{
    /// Create a new generic config wrapping the given data
    pub fn new(data: T) -> Self {
        Self {
            data,
            validation_rules: HashMap::new(),
            _phantom: PhantomData,
        }
    }

    /// Attach validation rules to this configuration
    pub fn with_validation(mut self, rules: HashMap<String, String>) -> Self {
        self.validation_rules = rules;
        self
    }
}

impl<T> BaseConfig for GenericConfig<T>
where
    T: Debug + Clone + Serialize + for<'de> Deserialize<'de> + Send + Sync,
{
    fn validate(&self) -> Result<(), String> {
        // Apply validation rules
        for (rule, message) in &self.validation_rules {
            // Simple validation framework - can be extended
            if rule == "required" && message.is_empty() {
                return Err(format!("Validation failed: {}", message));
            }
        }
        Ok(())
    }
}

/// Generic metadata wrapper that implements BaseMetadata
#[derive(Debug, Clone)]
pub struct GenericMetadata<T> {
    /// Unique identifier
    pub id: String,
    /// The wrapped metadata payload
    pub data: T,
    /// Monotonically increasing version number
    pub version: u64,
    /// Creation timestamp
    pub timestamp: DateTime<Utc>,
    /// Last modification timestamp
    pub updated_at: DateTime<Utc>,
    /// Arbitrary string tags for categorization
    pub tags: Vec<String>,
    /// Arbitrary JSON key-value properties
    pub properties: HashMap<String, serde_json::Value>,
}

impl<T> GenericMetadata<T>
where
    T: Debug + Clone + Serialize + for<'de> Deserialize<'de> + Send + Sync,
{
    /// Create metadata with the given ID and data, setting version to 1 and timestamps to now
    pub fn new(id: String, data: T) -> Self {
        let now = Utc::now();
        Self {
            id,
            data,
            version: 1,
            timestamp: now,
            updated_at: now,
            tags: Vec::new(),
            properties: HashMap::new(),
        }
    }

    /// Attach tags to this metadata entry
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    /// Attach properties to this metadata entry
    pub fn with_properties(mut self, properties: HashMap<String, serde_json::Value>) -> Self {
        self.properties = properties;
        self
    }
}

impl<T> BaseMetadata for GenericMetadata<T>
where
    T: Debug + Clone + Serialize + for<'de> Deserialize<'de> + Send + Sync,
{
    fn version(&self) -> u64 {
        self.version
    }

    fn id(&self) -> String {
        self.id.clone()
    }

    fn created_at(&self) -> DateTime<Utc> {
        self.timestamp
    }

    fn updated_at(&self) -> DateTime<Utc> {
        self.updated_at
    }
}

/// Generic statistics wrapper that implements BaseStats
#[derive(Debug, Clone)]
pub struct GenericStats<T> {
    /// The wrapped statistics payload
    pub data: T,
    /// When these statistics were last updated
    pub timestamp: DateTime<Utc>,
    /// Number of collections contributing to these statistics
    pub collection_count: u64,
    /// Number of times these statistics have been reset
    pub reset_count: u64,
}

impl<T> GenericStats<T>
where
    T: Debug + Clone + Serialize + for<'de> Deserialize<'de> + Send + Sync,
{
    /// Create a new statistics wrapper with a single collection count
    pub fn new(data: T) -> Self {
        Self {
            data,
            timestamp: Utc::now(),
            collection_count: 1,
            reset_count: 0,
        }
    }

    /// Replace the underlying statistics data and update the timestamp
    pub fn update_data(&mut self, data: T) {
        self.data = data;
        self.timestamp = Utc::now();
    }
}

impl<T> BaseStats for GenericStats<T>
where
    T: Debug + Clone + Serialize + for<'de> Deserialize<'de> + Send + Sync,
{
    fn aggregate(&mut self, other: &Self) {
        self.collection_count += other.collection_count;
        self.timestamp = Utc::now();
    }

    fn reset(&mut self) {
        self.collection_count = 0;
        self.reset_count += 1;
        self.timestamp = Utc::now();
    }

    fn timestamp(&self) -> DateTime<Utc> {
        self.timestamp
    }
}

/// Generic result wrapper that implements BaseResult
#[derive(Debug, Clone)]
pub struct GenericResult<T> {
    /// Whether the operation completed successfully
    pub success: bool,
    /// The result payload (present on success)
    pub data: Option<T>,
    /// Machine-readable error code (present on failure)
    pub error_code: Option<String>,
    /// Wall-clock processing time in microseconds
    pub processing_time_us: Option<u64>,
    /// Additional key-value metadata about the result
    pub metadata: HashMap<String, serde_json::Value>,
}

impl<T> GenericResult<T>
where
    T: Debug + Clone + Serialize + for<'de> Deserialize<'de> + Send + Sync,
{
    /// Create a successful result wrapping the given data
    pub fn success(data: T) -> Self {
        Self {
            success: true,
            data: Some(data),
            // error_message removed -  None,
            error_code: None,
            processing_time_us: None,
            metadata: HashMap::new(),
        }
    }

    /// Create a failed result with no data
    pub fn error() -> Self {
        Self {
            success: false,
            data: None,
            error_code: None,
            processing_time_us: None,
            metadata: HashMap::new(),
        }
    }

    /// Attach processing time metadata to this result
    pub fn with_processing_time(mut self, time_us: u64) -> Self {
        self.processing_time_us = Some(time_us);
        self
    }

    /// Attach an error code to this result
    pub fn with_error_code(mut self, code: String) -> Self {
        self.error_code = Some(code);
        self
    }
}

impl<T> BaseResult<T> for GenericResult<T>
where
    T: Debug + Clone + Serialize + for<'de> Deserialize<'de> + Send + Sync,
{
    fn is_success(&self) -> bool {
        self.success
    }

    fn data(&self) -> Option<&T> {
        self.data.as_ref()
    }

    fn error(&self) -> Option<&str> {
        self.error_code.as_deref()
    }

    fn processing_time_us(&self) -> Option<u64> {
        self.processing_time_us
    }
}

// Manual Serialize/Deserialize implementations for generic types
impl<T> Serialize for GenericConfig<T>
where
    T: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serialize_fields!(serializer, "GenericConfig",
            "data" => self.data,
            "validation_rules" => self.validation_rules,
        )
    }
}

impl<'de, T> Deserialize<'de> for GenericConfig<T>
where
    T: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct GenericConfigVisitor<T>(PhantomData<T>);

        impl<'de, T> Visitor<'de> for GenericConfigVisitor<T>
        where
            T: Deserialize<'de>,
        {
            type Value = GenericConfig<T>;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct GenericConfig")
            }

            fn visit_map<V>(self, mut map: V) -> Result<GenericConfig<T>, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut data = None;
                let mut validation_rules = None;

                while let Some(key) = map.next_key::<String>()? {
                    match key.as_str() {
                        "data" => take_field!(map, data, "data"),
                        "validation_rules" => {
                            take_field!(map, validation_rules, "validation_rules")
                        }
                        _ => {
                            let _ = map.next_value::<serde_json::Value>()?;
                        }
                    }
                }

                let data = data.ok_or_else(|| de::Error::missing_field("data"))?;
                let validation_rules = validation_rules.unwrap_or_else(HashMap::new);

                Ok(GenericConfig {
                    data,
                    validation_rules,
                    _phantom: PhantomData,
                })
            }
        }

        deserializer.deserialize_map(GenericConfigVisitor(PhantomData))
    }
}

impl<T> Serialize for GenericMetadata<T>
where
    T: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serialize_fields!(serializer, "GenericMetadata",
            "id" => self.id,
            "data" => self.data,
            "version" => self.version,
            "timestamp" => self.timestamp,
            "updated_at" => self.updated_at,
            "tags" => self.tags,
            "properties" => self.properties,
        )
    }
}

impl<'de, T> Deserialize<'de> for GenericMetadata<T>
where
    T: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "snake_case")]
        enum Field {
            Id,
            Data,
            Version,
            Timestamp,
            UpdatedAt,
            Tags,
            Properties,
        }

        struct GenericMetadataVisitor<T>(PhantomData<T>);

        impl<'de, T> Visitor<'de> for GenericMetadataVisitor<T>
        where
            T: Deserialize<'de>,
        {
            type Value = GenericMetadata<T>;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct GenericMetadata")
            }

            fn visit_map<V>(self, mut map: V) -> Result<GenericMetadata<T>, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut id = None;
                let mut data = None;
                let mut version = None;
                let mut timestamp = None;
                let mut updated_at = None;
                let mut tags = None;
                let mut properties = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Id => take_field!(map, id, "id"),
                        Field::Data => take_field!(map, data, "data"),
                        Field::Version => take_field!(map, version, "version"),
                        Field::Timestamp => take_field!(map, timestamp, "timestamp"),
                        Field::UpdatedAt => take_field!(map, updated_at, "updated_at"),
                        Field::Tags => take_field!(map, tags, "tags"),
                        Field::Properties => take_field!(map, properties, "properties"),
                    }
                }

                let id = id.ok_or_else(|| de::Error::missing_field("id"))?;
                let data = data.ok_or_else(|| de::Error::missing_field("data"))?;
                let version = version.unwrap_or(1);
                let timestamp = timestamp.ok_or_else(|| de::Error::missing_field("timestamp"))?;
                let updated_at = updated_at.unwrap_or(timestamp);
                let tags = tags.unwrap_or_else(Vec::new);
                let properties = properties.unwrap_or_else(HashMap::new);

                Ok(GenericMetadata {
                    id,
                    data,
                    version,
                    timestamp,
                    updated_at,
                    tags,
                    properties,
                })
            }
        }

        deserializer.deserialize_map(GenericMetadataVisitor(PhantomData))
    }
}

impl<T> Serialize for GenericStats<T>
where
    T: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serialize_fields!(serializer, "GenericStats",
            "data" => self.data,
            "timestamp" => self.timestamp,
            "collection_count" => self.collection_count,
            "reset_count" => self.reset_count,
        )
    }
}

impl<'de, T> Deserialize<'de> for GenericStats<T>
where
    T: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "snake_case")]
        enum Field {
            Data,
            Timestamp,
            CollectionCount,
            ResetCount,
        }

        struct GenericStatsVisitor<T>(PhantomData<T>);

        impl<'de, T> Visitor<'de> for GenericStatsVisitor<T>
        where
            T: Deserialize<'de>,
        {
            type Value = GenericStats<T>;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct GenericStats")
            }

            fn visit_map<V>(self, mut map: V) -> Result<GenericStats<T>, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut data = None;
                let mut timestamp = None;
                let mut collection_count = None;
                let mut reset_count = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Data => take_field!(map, data, "data"),
                        Field::Timestamp => take_field!(map, timestamp, "timestamp"),
                        Field::CollectionCount => {
                            take_field!(map, collection_count, "collection_count")
                        }
                        Field::ResetCount => take_field!(map, reset_count, "reset_count"),
                    }
                }

                let data = data.ok_or_else(|| de::Error::missing_field("data"))?;
                let timestamp = timestamp.ok_or_else(|| de::Error::missing_field("timestamp"))?;
                let collection_count = collection_count.unwrap_or(0);
                let reset_count = reset_count.unwrap_or(0);

                Ok(GenericStats {
                    data,
                    timestamp,
                    collection_count,
                    reset_count,
                })
            }
        }

        deserializer.deserialize_map(GenericStatsVisitor(PhantomData))
    }
}

impl<T> Serialize for GenericResult<T>
where
    T: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serialize_fields!(serializer, "GenericResult",
            "success" => self.success,
            "data" => self.data,
            "error_code" => self.error_code,
            "processing_time_us" => self.processing_time_us,
            "metadata" => self.metadata,
        )
    }
}

impl<'de, T> Deserialize<'de> for GenericResult<T>
where
    T: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "snake_case")]
        enum Field {
            Success,
            Data,
            ErrorCode,
            ProcessingTimeUs,
            Metadata,
        }

        struct GenericResultVisitor<T>(PhantomData<T>);

        impl<'de, T> Visitor<'de> for GenericResultVisitor<T>
        where
            T: Deserialize<'de>,
        {
            type Value = GenericResult<T>;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct GenericResult")
            }

            fn visit_map<V>(self, mut map: V) -> Result<GenericResult<T>, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut success = None;
                let mut data = None;
                let mut error_code = None;
                let mut processing_time_us = None;
                let mut metadata = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Success => take_field!(map, success, "success"),
                        Field::Data => take_field!(map, data, "data"),
                        Field::ErrorCode => take_field!(map, error_code, "error_code"),
                        Field::ProcessingTimeUs => {
                            take_field!(map, processing_time_us, "processing_time_us")
                        }
                        Field::Metadata => take_field!(map, metadata, "metadata"),
                    }
                }

                let success = success.ok_or_else(|| de::Error::missing_field("success"))?;
                let data = data.unwrap_or(None);
                let error_code = error_code.unwrap_or(None);
                let processing_time_us = processing_time_us.unwrap_or(None);
                let metadata = metadata.unwrap_or_else(HashMap::new);

                Ok(GenericResult {
                    success,
                    data,
                    error_code,
                    processing_time_us,
                    metadata,
                })
            }
        }

        deserializer.deserialize_map(GenericResultVisitor(PhantomData))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generic_config_validates_defaults_and_round_trips_json() {
        let config = GenericConfig::new(serde_json::json!({"mode": "test"}));
        assert!(config.validate().is_ok());

        let mut rules = HashMap::new();
        rules.insert("required".to_string(), String::new());
        let invalid = config.clone().with_validation(rules);
        assert_eq!(invalid.validate().unwrap_err(), "Validation failed: ");

        let encoded = serde_json::to_string(&config).unwrap();
        let decoded: GenericConfig<serde_json::Value> = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded.data, serde_json::json!({"mode": "test"}));
        assert!(decoded.validation_rules.is_empty());

        let with_unknown: GenericConfig<u32> =
            serde_json::from_str(r#"{"data":7,"extra":true}"#).unwrap();
        assert_eq!(with_unknown.data, 7);
        assert!(serde_json::from_str::<GenericConfig<u32>>(r#"{"validation_rules":{}}"#).is_err());
        assert!(serde_json::from_str::<GenericConfig<u32>>(r#"{"data":1,"data":2}"#).is_err());
    }

    #[test]
    fn generic_metadata_exposes_base_metadata_and_manual_serde_defaults() {
        let metadata = GenericMetadata::new("meta-1".to_string(), 42_u32)
            .with_tags(vec!["hot".to_string()])
            .with_properties(HashMap::from([(
                "owner".to_string(),
                serde_json::json!("alice"),
            )]));

        assert_eq!(metadata.id(), "meta-1");
        assert_eq!(metadata.version(), 1);
        assert_eq!(metadata.created_at(), metadata.timestamp);
        assert_eq!(metadata.updated_at(), metadata.updated_at);
        assert_eq!(metadata.tags, vec!["hot"]);
        assert_eq!(metadata.properties["owner"], serde_json::json!("alice"));

        let encoded = serde_json::to_string(&metadata).unwrap();
        let decoded: GenericMetadata<u32> = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded.id, "meta-1");
        assert_eq!(decoded.data, 42);
        assert_eq!(decoded.version, 1);
        assert_eq!(decoded.tags, vec!["hot"]);

        let minimal = serde_json::json!({
            "id": "meta-2",
            "data": 11,
            "timestamp": metadata.timestamp,
        });
        let decoded_minimal: GenericMetadata<u32> = serde_json::from_value(minimal).unwrap();
        assert_eq!(decoded_minimal.version, 1);
        assert_eq!(decoded_minimal.updated_at, decoded_minimal.timestamp);
        assert!(decoded_minimal.tags.is_empty());
        assert!(decoded_minimal.properties.is_empty());

        assert!(serde_json::from_str::<GenericMetadata<u32>>(r#"{"data":1}"#).is_err());
        assert!(
            serde_json::from_str::<GenericMetadata<u32>>(&format!(
                r#"{{"id":"a","id":"b","data":1,"timestamp":"{}"}}"#,
                metadata.timestamp.to_rfc3339()
            ))
            .is_err()
        );
    }

    #[test]
    fn generic_stats_aggregate_reset_update_and_round_trip() {
        let mut stats = GenericStats::new(10_u32);
        assert_eq!(stats.collection_count, 1);
        assert_eq!(stats.reset_count, 0);
        assert_eq!(stats.timestamp(), stats.timestamp);

        stats.update_data(20);
        assert_eq!(stats.data, 20);
        stats.aggregate(&GenericStats {
            data: 99,
            timestamp: Utc::now(),
            collection_count: 4,
            reset_count: 0,
        });
        assert_eq!(stats.collection_count, 5);
        stats.reset();
        assert_eq!(stats.collection_count, 0);
        assert_eq!(stats.reset_count, 1);

        let encoded = serde_json::to_string(&stats).unwrap();
        let decoded: GenericStats<u32> = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded.data, 20);
        assert_eq!(decoded.collection_count, 0);
        assert_eq!(decoded.reset_count, 1);

        let minimal = serde_json::json!({
            "data": 5,
            "timestamp": Utc::now(),
        });
        let decoded_minimal: GenericStats<u32> = serde_json::from_value(minimal).unwrap();
        assert_eq!(decoded_minimal.collection_count, 0);
        assert_eq!(decoded_minimal.reset_count, 0);

        assert!(serde_json::from_str::<GenericStats<u32>>(r#"{"collection_count":1}"#).is_err());
        assert!(
            serde_json::from_str::<GenericStats<u32>>(&format!(
                r#"{{"data":1,"timestamp":"{}","reset_count":1,"reset_count":2}}"#,
                Utc::now().to_rfc3339()
            ))
            .is_err()
        );
    }

    #[test]
    fn generic_result_success_error_metadata_and_manual_serde_defaults() {
        let success = GenericResult::success("ok".to_string()).with_processing_time(33);
        assert!(success.is_success());
        assert_eq!(success.data().map(String::as_str), Some("ok"));
        assert_eq!(success.error(), None);
        assert_eq!(success.processing_time_us(), Some(33));

        let error: GenericResult<String> =
            GenericResult::error().with_error_code("E_BAD".to_string());
        assert!(!error.is_success());
        assert_eq!(error.data(), None);
        assert_eq!(error.error(), Some("E_BAD"));

        let encoded = serde_json::to_string(&success).unwrap();
        let decoded: GenericResult<String> = serde_json::from_str(&encoded).unwrap();
        assert!(decoded.success);
        assert_eq!(decoded.data.as_deref(), Some("ok"));
        assert_eq!(decoded.processing_time_us, Some(33));
        assert!(decoded.metadata.is_empty());

        let minimal: GenericResult<u32> = serde_json::from_str(r#"{"success":false}"#).unwrap();
        assert!(!minimal.success);
        assert_eq!(minimal.data, None);
        assert_eq!(minimal.error_code, None);
        assert_eq!(minimal.processing_time_us, None);
        assert!(minimal.metadata.is_empty());

        assert!(serde_json::from_str::<GenericResult<u32>>(r#"{"data":1}"#).is_err());
        assert!(
            serde_json::from_str::<GenericResult<u32>>(r#"{"success":true,"success":false}"#)
                .is_err()
        );
    }
}
