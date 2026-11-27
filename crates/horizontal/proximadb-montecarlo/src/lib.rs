// Copyright (C) 2025 ProximaDB
// SPDX-License-Identifier: Apache-2.0
//! Monte-Carlo option-pricing kernel, extracted from the root `compute`
//! module (TD-DECOMP-41).
//!
//! [`montecarlo`] ships a dependency-free Black-Scholes pricer plus a
//! parallel ([`montecarlo::mc_price_batch_seq`] / rayon) Monte-Carlo path
//! simulator for European option pricing. It mirrors how Spark prices
//! options as a UDF and depends only on `rayon`, keeping it a clean
//! horizontal-tier leaf.

// Mirrors the root crate's crate-level allow (src/lib.rs:27): the Monte-Carlo
// pricer entry points carry several scalar parameters pending a config-struct
// refactor. Behavior-neutral lint posture, carried verbatim with the code.
#![allow(clippy::too_many_arguments)]

pub mod montecarlo;
