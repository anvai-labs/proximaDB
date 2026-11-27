// Copyright (C) 2025 ProximaDB
// SPDX-License-Identifier: Apache-2.0
//! Kafka consumer configuration types, extracted from the root
//! `streaming/kafka` module (TD-DECOMP-40).
//!
//! [`config`] carries the serde-serialisable settings for the Kafka vector
//! ingestion consumer: broker endpoints, consumer-group config, commit
//! strategy, dead-letter-queue (DLQ) policy, and deserialization format
//! ([`config::KafkaConsumerConfig`] and friends). The module depends only on
//! `serde` — the heavy `rdkafka` consumer/deserializer impls stay
//! root-side, keeping this a clean horizontal-tier config leaf.

pub mod config;
