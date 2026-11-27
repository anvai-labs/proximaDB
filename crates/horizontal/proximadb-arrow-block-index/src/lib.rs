// Copyright (C) 2025 ProximaDB
// SPDX-License-Identifier: Apache-2.0
//! Arrow block secondary index, extracted from the root
//! `storage/.../arrow_block` module (TD-DECOMP-58).
//!
//! [`index`] provides [`index::ArrowBlockIndex`] + [`index::ArrowIndexEntry`]
//! for block-level secondary indexing. Depends only on `serde` + `bincode`,
//! keeping it a clean horizontal-tier leaf.

pub mod index;
