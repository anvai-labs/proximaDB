// Copyright (C) 2025 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! Neutral usage-event emission (ADR-067 / TD-SANDHI-1 embeddings; TD-SANDHI-2 generation LLM).
//!
//! Serializes to the AnvaiOps/`sandhi` `usage-event.v1` schema — the neutral, cross-repo
//! boundary object emitted once per external model call. **Measured units only: NO dollars,
//! NO tier/SKU names** (OSS emits units, AnvaiOps prices — ADR-067 / ADR-060).
//!
//! Transport is the ADR-067 "in-process emit": a direct crate call that logs the event through
//! `tracing` (operators ship logs to their sink of record). It **complements** the KEU meter
//! (`consumption_metrics`), never replaces it (AnvaiOps ADR-0020 D6), and is **default-inert** —
//! nothing is emitted unless `PROXIMADB_EMIT_USAGE_EVENTS` is set truthy. Local BGE ONNX has no
//! external egress and does not emit (it keeps its compute-based KEU accrual).
//!
//! Dependency direction is one-way (ProximaDB → `sandhi`, ADR-060): the event *struct* mirrors the
//! wire schema locally (embedding path, TD-SANDHI-1), and the generation path additionally adopts
//! `sandhi-core`'s single-sourced, fixture-proven usage *parsers* (TD-SANDHI-2 / ADR-0047 D10a) —
//! rather than hand-rolling a per-provider extractor.
//!
//! Measurement provenance (TD-SANDHI-3): both emit paths stamp `duration_ms` (adapter-boundary
//! wall clock — which per the schema **excludes host-side queueing** — measured by the caller),
//! `usage_completeness` (`Final` — every emitted event carries a complete accounting for its
//! logical call), `usage_basis` (`ProviderReported` vs `Estimated` — the embedding heuristic
//! count×512 path is the one estimated basis), and `outcome` (`success` — emit sites are
//! success-path only, and an omitted `outcome` decodes to no default, unlike the other
//! measurement fields). On the **generation** path, bodies that fail to parse or parse to
//! all-zero usage emit nothing (a parsed zero is no measurement); on the **embedding** path an
//! absent/zero provider usage falls back to the heuristic with an `Estimated` basis, and a
//! measured batch clamps each record-carrying payload's share to a minimum of 1 token
//! (`measured_share` in the drainer) — never certifying a zero, which on the wire would be
//! indistinguishable from the gateway-bug zeros dropped above.
//! `time_to_first_token_ms` stays omitted: ProximaDB's LLM integration is non-streaming only,
//! and the field is streams-only by schema.

use serde::Serialize;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

/// The `tracing` target external tooling filters on to collect neutral usage events.
pub const USAGE_EVENT_TARGET: &str = "proximadb::usage_event";

/// Env flag gating emission. Unset/false ⇒ fully inert (no allocation, no log).
const EMIT_FLAG_ENV: &str = "PROXIMADB_EMIT_USAGE_EVENTS";

/// Backend classification (schema `backend` enum).
pub const BACKEND_EXTERNAL: &str = "external";
/// Self-hosted backend (local inference — Ollama / vLLM). Tokens are display-only there; the
/// cost basis is GPU-hours (AnvaiOps ADR-0020 D4), which ProximaDB does not measure here.
pub const BACKEND_SELF_HOSTED: &str = "self_hosted";

// Re-exported so emit-site callers stamp provenance through the crate seam (SOLID: stable crate
// path, not the `sandhi_core` internal path).
pub use sandhi_core::UsageBasis;
pub use sandhi_core::UsageCompleteness;

fn emission_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var(EMIT_FLAG_ENV)
            .map(|v| {
                matches!(
                    v.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "on"
                )
            })
            .unwrap_or(false)
    })
}

fn next_request_id(kind: &str) -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let millis = chrono::Utc::now().timestamp_millis();
    format!("keu-{kind}-{millis}-{n}")
}

/// The neutral usage event — `sandhi` `usage-event.v1` (synced to the sandhi-core 0.1.5 schema).
/// `additionalProperties:false`, so this struct carries exactly the schema's fields. Optional
/// fields are omitted when `None` (a valid encoding of the nullable/defaulted schema fields);
/// every non-required field here is `None` by default, so events that set none of them serialize
/// byte-identically to the pre-sync emission.
#[derive(Debug, Clone, Serialize)]
pub struct UsageEvent {
    pub schema_version: &'static str,
    pub request_id: String,
    /// RFC 3339, set when the usage was finalized.
    pub occurred_at: String,
    /// Neutral provider slug (`azure_openai` / `openai` / `cohere` / `byo`). Never a tier/SKU.
    pub provider: String,
    /// Model id as sent to the provider.
    pub model: String,
    /// `external` for provider APIs (billed in tokens).
    pub backend: &'static str,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub virtual_key_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subject_id: Option<String>,
    /// The team/tenant the call is attributed to. ProximaDB maps its `tenant_id` here (the drainer
    /// knows the tenant/account, not the individual user).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub group_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub route: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,

    // --- ADR-0005 D7 neutral identity (attribution metadata, never pricing; sandhi-core 0.1.5) ---
    /// Caller-supplied key for at-most-once semantics across retries of one logical call.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub idempotency_key: Option<String>,
    /// Agent-run identifier; groups every call one run makes (cost-tree root).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub run_id: Option<String>,
    /// Step within a run; child dimension under `run_id`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub step_id: Option<String>,
    /// Parent step/run for nested agents, so an agent's cost tree is reconstructable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_id: Option<String>,
    /// W3C `traceparent` value, linking the event into distributed traces.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trace_context: Option<String>,

    /// Fresh (non-cached) input tokens — the provider's real usage count.
    pub tokens_in: u64,
    /// Completion tokens. Always 0 for embeddings (they emit vectors, not completion tokens).
    pub tokens_out: u64,
    pub cache_creation_tokens: u64,
    pub cache_read_tokens: u64,

    // --- Measurement provenance (schema defaults apply when omitted: `unavailable` /
    // `provider_reported` / 1 attempt). `None` ⇒ key omitted, so pre-sync events stay
    // byte-identical. The enum types are consumed from `sandhi-core` (serde-only dep, ADR-060)
    // rather than re-mirrored, so their wire spellings can never drift. ---
    /// Whether token counts are final, partial, or unavailable for this logical call.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage_completeness: Option<sandhi_core::UsageCompleteness>,
    /// Whether counts were provider-measured or byte-estimated (Sandhi TD-0013 P3).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage_basis: Option<sandhi_core::UsageBasis>,
    /// Number of upstream attempts made for this logical call.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attempts: Option<u32>,
    /// Stable terminal outcome such as `success`, `error`, or `cancelled`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub outcome: Option<String>,
    /// Provider-supplied request identifier when one was returned.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub upstream_request_id: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_seconds: Option<f64>,

    /// Wall-clock duration of the logical call in milliseconds, measured at the adapter boundary
    /// (includes retries; excludes host-side queueing — sandhi-core `usage-event.v1`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u64>,
    /// Streams only: milliseconds from request start to the first delivered item.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub time_to_first_token_ms: Option<u64>,
    /// Reasoning tokens when the provider reports them separately (OpenAI `reasoning_tokens`,
    /// Gemini `thoughtsTokenCount`). Absent when folded into `tokens_out` or not reported.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u64>,
}

impl UsageEvent {
    /// Build the event for one external **embedding** call, metered from the provider's real
    /// input-token count. `route` is the operation label (e.g. `"embed_batch"`).
    pub fn external_embedding(
        provider: impl Into<String>,
        model: impl Into<String>,
        tenant_id: Option<&str>,
        tokens_in: u64,
        route: impl Into<String>,
    ) -> Self {
        Self {
            schema_version: "1",
            request_id: next_request_id("emb"),
            occurred_at: chrono::Utc::now().to_rfc3339(),
            provider: provider.into(),
            model: model.into(),
            backend: BACKEND_EXTERNAL,
            virtual_key_id: None,
            subject_id: None,
            group_id: tenant_id.map(str::to_string),
            route: Some(route.into()),
            session_id: None,
            idempotency_key: None,
            run_id: None,
            step_id: None,
            parent_id: None,
            trace_context: None,
            tokens_in,
            tokens_out: 0,
            cache_creation_tokens: 0,
            cache_read_tokens: 0,
            usage_completeness: None,
            usage_basis: None,
            attempts: None,
            outcome: None,
            upstream_request_id: None,
            gpu_seconds: None,
            duration_ms: None,
            time_to_first_token_ms: None,
            reasoning_tokens: None,
        }
    }

    /// Build the event for one **generation-LLM** call, with the full token breakdown incl. the
    /// prompt-cache split (TD-SANDHI-2 / ADR-0047 D10a). `backend` is `BACKEND_EXTERNAL` for
    /// provider APIs or `BACKEND_SELF_HOSTED` for local inference; `route` is the operation label.
    #[allow(clippy::too_many_arguments)]
    pub fn generation_llm(
        provider: impl Into<String>,
        model: impl Into<String>,
        tenant_id: Option<&str>,
        tokens_in: u64,
        tokens_out: u64,
        cache_creation_tokens: u64,
        cache_read_tokens: u64,
        route: impl Into<String>,
        backend: &'static str,
    ) -> Self {
        Self {
            schema_version: "1",
            request_id: next_request_id("gen"),
            occurred_at: chrono::Utc::now().to_rfc3339(),
            provider: provider.into(),
            model: model.into(),
            backend,
            virtual_key_id: None,
            subject_id: None,
            group_id: tenant_id.map(str::to_string),
            route: Some(route.into()),
            session_id: None,
            idempotency_key: None,
            run_id: None,
            step_id: None,
            parent_id: None,
            trace_context: None,
            tokens_in,
            tokens_out,
            cache_creation_tokens,
            cache_read_tokens,
            usage_completeness: None,
            usage_basis: None,
            attempts: None,
            outcome: None,
            upstream_request_id: None,
            gpu_seconds: None,
            duration_ms: None,
            time_to_first_token_ms: None,
            reasoning_tokens: None,
        }
    }

    /// Stamp measurement provenance onto an embedding event (TD-SANDHI-3): the adapter-boundary
    /// wall clock of the logical call (which per the schema contract **excludes host-side
    /// queueing** — measure around the provider call, not the queue submit), and whether its
    /// counts were `ProviderReported` (the provider returned a real usage figure) or `Estimated`
    /// (ProximaDB's own heuristic — record-count × 512 — because the provider reports no usable
    /// usage). Completeness is `Final` (a stamped event accounts for a completed call, never a
    /// partial one), and the outcome is `success` — callers invoke this only on the success path,
    /// and the reference producer always stamps an outcome (an omitted `outcome` decodes to no
    /// default, so a `outcome == "success"` filter would drop the event).
    ///
    /// `duration_ms` is `Option`: a multi-tenant batch emits one event per payload but the
    /// batch's wall clock belongs to exactly one designated carrier event (N events × full
    /// duration would inflate any `SUM(duration_ms)` rollup N×); `None` omits the key while
    /// still stamping basis/completeness/outcome.
    ///
    /// Named `with_provenance`, NOT `with_measurement`: sandhi-core's own
    /// `UsageEvent::with_measurement(completeness, attempts, outcome, upstream_request_id)` sets
    /// different fields — the near-collision on this mirrored type invited exactly that mixup.
    #[must_use]
    pub fn with_provenance(
        mut self,
        usage_basis: sandhi_core::UsageBasis,
        duration_ms: Option<u64>,
    ) -> Self {
        self.usage_basis = Some(usage_basis);
        self.usage_completeness = Some(sandhi_core::UsageCompleteness::Final);
        self.duration_ms = duration_ms;
        self.outcome = Some("success".to_string());
        self
    }

    /// Emit the event through `tracing` — **best-effort, off the hot path, default-inert**. No-op
    /// unless `PROXIMADB_EMIT_USAGE_EVENTS` is truthy. Never fails the caller.
    pub fn emit(&self) {
        if !emission_enabled() {
            return;
        }
        match serde_json::to_string(self) {
            Ok(json) => tracing::info!(target: USAGE_EVENT_TARGET, usage_event = %json),
            Err(e) => tracing::debug!("usage-event serialize failed (ignored): {e}"),
        }
    }
}

/// Build (without emitting) the neutral usage event for one generation-LLM call, **adopting the
/// fixture-proven `sandhi-core` usage parser** for `provider` (single-sourced metering trust —
/// AnvaiOps ADR-0047 D10a / Sandhi ADR-0003). Returns `None` if the body doesn't parse, carries
/// no usage, carries inconsistent extended usage counts, or parses to all-zero usage
/// (no billable signal — fabricating a zero-token event
/// would be "a silent, wrong measurement", per sandhi-core's own Cohere hardening). Split out
/// from [`emit_generation_usage`] so the parser-dispatch + backend mapping is unit-testable
/// without the env-gated emit.
fn build_generation_event(
    provider: &str,
    model: &str,
    tenant_id: Option<&str>,
    raw_response_body: &str,
    route: &str,
    duration_ms: u64,
) -> Option<UsageEvent> {
    let value: serde_json::Value = serde_json::from_str(raw_response_body).ok()?;
    use sandhi_core::EndpointFamilyV1;
    use sandhi_core::usage::{
        parse_anthropic_usage, parse_cohere_usage, parse_ollama_usage, parse_openai_usage,
        qualify_buffered_usage, validate_openai_buffered_usage,
    };
    // Qualify Sandhi's extended counters when present, before marking usage final.
    let usage = match provider {
        "anthropic" => parse_anthropic_usage(&value).filter(|usage| {
            value
                .pointer("/usage/output_tokens_details/thinking_tokens")
                .is_none_or(|thinking| {
                    thinking.as_u64() == Some(usage.reasoning_tokens)
                        && usage.reasoning_tokens <= usage.tokens_out
                })
        }),
        "cohere" => parse_cohere_usage(&value),
        "ollama" => parse_ollama_usage(&value).filter(|_| {
            value.get("prompt_eval_cached_count").is_none()
                || qualify_buffered_usage(EndpointFamilyV1::OllamaChat, &value).is_ok()
        }),
        _ => parse_openai_usage(&value).filter(|_| {
            value
                .pointer("/usage/prompt_tokens_details/cache_write_tokens")
                .is_none()
                || validate_openai_buffered_usage(&value).is_ok()
        }),
    }?;
    // All-zero usage is no measurement (TD-SANDHI-3 adversarial review): empty objects
    // (`{"usage":{}}`), `null`, string-typed numbers, and explicit zeros (Ollama reports
    // `prompt_eval_count: 0` on cached prompts) all land here. sandhi-core guards this in its
    // Cohere parser; we guard at the dispatch so every arm gets it. Strictly stricter than
    // sandhi's own producers (which stamp `Final` zeros) — dropping the event only loses a
    // zero-token billing signal, while certifying it as final+provider-reported would poison the
    // very trust filter `usage_completeness` exists to enable.
    if usage.tokens_in == 0
        && usage.tokens_out == 0
        && usage.cache_creation_tokens == 0
        && usage.cache_read_tokens == 0
        && usage.reasoning_tokens == 0
    {
        return None;
    }
    let backend = match provider {
        "ollama" | "vllm" => BACKEND_SELF_HOSTED,
        _ => BACKEND_EXTERNAL,
    };
    let mut event = UsageEvent::generation_llm(
        provider,
        model,
        tenant_id,
        usage.tokens_in,
        usage.tokens_out,
        usage.cache_creation_tokens,
        usage.cache_read_tokens,
        route,
        backend,
    );
    // sandhi-core 0.1.5 parsers surface separately-reported reasoning tokens (e.g. OpenAI
    // o-series). Present ⇒ carried; otherwise omitted (not zero), same as sandhi's own
    // `ParsedUsage` → event mapping, so non-reasoning events keep the pre-sync shape.
    event.reasoning_tokens = (usage.reasoning_tokens > 0).then_some(usage.reasoning_tokens);
    // Measurement provenance (TD-SANDHI-3), mirroring sandhi-core `ParsedUsage::apply`'s
    // fill-if-empty semantics: a body whose usage parsed to a nonzero count is a complete,
    // provider-measured accounting for the call (all-zero/failed parses emit no event at all, so
    // `Partial`/`Unavailable` never apply here), the duration is the provider adapter's wall
    // clock, and the outcome is stamped because emit sites are success-path only — the reference
    // producer always sets one, and an omitted `outcome` decodes to no default.
    event.usage_completeness = Some(sandhi_core::UsageCompleteness::Final);
    event.usage_basis = Some(sandhi_core::UsageBasis::ProviderReported);
    event.duration_ms = Some(duration_ms);
    event.outcome = Some("success".to_string());
    Some(event)
}

/// Emit the neutral usage event for one **generation-LLM** call. `provider` is the neutral slug
/// (`anthropic` / `openai` / `azure_openai` / `cohere` / `ollama` / `vllm` / `huggingface`).
/// `duration_ms` is the adapter-boundary wall clock of the call (the provider's measured
/// `response_time_ms`). **Best-effort + default-inert**: no parse work and no emit unless
/// `PROXIMADB_EMIT_USAGE_EVENTS` is set; never raises into the caller (the provider hot path).
pub fn emit_generation_usage(
    provider: &str,
    model: &str,
    tenant_id: Option<&str>,
    raw_response_body: &str,
    route: &str,
    duration_ms: u64,
) {
    if !emission_enabled() {
        return;
    }
    if let Some(ev) = build_generation_event(
        provider,
        model,
        tenant_id,
        raw_response_body,
        route,
        duration_ms,
    ) {
        ev.emit();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serializes_to_neutral_schema_shape() {
        let ev = UsageEvent::external_embedding(
            "azure_openai",
            "text-embedding-3-small",
            Some("tenant-abc"),
            1234,
            "embed_batch",
        );
        let v: serde_json::Value = serde_json::to_value(&ev).unwrap();
        assert_eq!(v["schema_version"], "1");
        assert_eq!(v["backend"], "external");
        assert_eq!(v["provider"], "azure_openai");
        assert_eq!(v["tokens_in"], 1234);
        assert_eq!(v["tokens_out"], 0);
        assert_eq!(v["group_id"], "tenant-abc");
        assert_eq!(v["route"], "embed_batch");
        // Neutral contract: units only, never dollars/tiers.
        assert!(v.get("cost").is_none() && v.get("usd").is_none() && v.get("tier").is_none());
        // Omitted optional attribution is absent (a valid nullable encoding).
        assert!(v.get("subject_id").is_none());
        assert!(v["request_id"].as_str().unwrap().starts_with("keu-emb-"));
    }

    #[test]
    fn generation_event_anthropic_carries_cache_split() {
        // Adopts sandhi-core::parse_anthropic_usage — the cache split the provider DTO dropped.
        let body = serde_json::json!({
            "model": "claude-3-5-sonnet",
            "content": [{ "type": "text", "text": "hi" }],
            "usage": {
                "input_tokens": 100, "output_tokens": 20,
                "cache_creation_input_tokens": 30, "cache_read_input_tokens": 40,
                "output_tokens_details": { "thinking_tokens": 8 }
            }
        })
        .to_string();
        let ev = build_generation_event(
            "anthropic",
            "claude-3-5-sonnet",
            Some("tenant-x"),
            &body,
            "query",
            4321,
        )
        .unwrap();
        let v = serde_json::to_value(&ev).unwrap();
        assert_eq!(v["provider"], "anthropic");
        assert_eq!(v["backend"], "external");
        assert_eq!(v["tokens_in"], 100);
        assert_eq!(v["tokens_out"], 20);
        assert_eq!(v["reasoning_tokens"], 8);
        assert_eq!(v["cache_creation_tokens"], 30);
        assert_eq!(v["cache_read_tokens"], 40);
        assert_eq!(v["group_id"], "tenant-x");
        assert!(v["request_id"].as_str().unwrap().starts_with("keu-gen-"));
        // TD-SANDHI-3: a parsed body is a complete, provider-measured accounting.
        assert_eq!(v["usage_completeness"], "final");
        assert_eq!(v["usage_basis"], "provider_reported");
        assert_eq!(v["duration_ms"], 4321);
        assert_eq!(v["outcome"], "success");
        // Streams-only field stays omitted on the non-streaming integration.
        assert!(v.get("time_to_first_token_ms").is_none());
    }

    #[test]
    fn generation_event_openai_family_and_selfhosted_backend() {
        let body = serde_json::json!({ "usage": { "prompt_tokens": 50, "completion_tokens": 10 } })
            .to_string();
        // OpenAI-compat family (default arm) → external.
        let ext =
            build_generation_event("azure_openai", "gpt-4o", None, &body, "query", 10).unwrap();
        assert_eq!(serde_json::to_value(&ext).unwrap()["backend"], "external");
        // vLLM parses via the same OpenAI shape but classifies as self_hosted.
        let sh = build_generation_event("vllm", "llama", None, &body, "query", 10).unwrap();
        let v = serde_json::to_value(&sh).unwrap();
        assert_eq!(v["backend"], "self_hosted");
        assert_eq!(v["tokens_in"], 50);
        assert_eq!(v["cache_creation_tokens"], 0);
    }

    #[test]
    fn generation_event_openai_separates_cache_writes_from_fresh_input() {
        let body = serde_json::json!({
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "prompt_tokens_details": { "cached_tokens": 30, "cache_write_tokens": 10 }
            }
        })
        .to_string();
        let event = build_generation_event("openai", "m", None, &body, "query", 10).unwrap();
        assert_eq!(event.tokens_in, 60);
        assert_eq!(event.tokens_out, 20);
        assert_eq!(event.cache_read_tokens, 30);
        assert_eq!(event.cache_creation_tokens, 10);
    }

    #[test]
    fn generation_event_ollama_separates_cache_reads_from_fresh_input() {
        let body = r#"{"done": true, "prompt_eval_count": 100, "prompt_eval_cached_count": 30, "eval_count": 20}"#;
        let event = build_generation_event("ollama", "m", None, body, "query", 10).unwrap();
        assert_eq!(event.tokens_in, 70);
        assert_eq!(event.tokens_out, 20);
        assert_eq!(event.cache_read_tokens, 30);
    }

    #[test]
    fn generation_event_rejects_inconsistent_extended_usage_counts() {
        let cases = [
            (
                "openai",
                serde_json::json!({ "usage": {
                "prompt_tokens": 10, "completion_tokens": 1,
                "prompt_tokens_details": { "cached_tokens": 7, "cache_write_tokens": 4 }
            } }),
            ),
            (
                "ollama",
                serde_json::json!({ "done": true, "prompt_eval_count": 0, "eval_count": 0,
                "prompt_eval_cached_count": 75 }),
            ),
            (
                "anthropic",
                serde_json::json!({ "usage": { "input_tokens": 12, "output_tokens": 2,
                "output_tokens_details": { "thinking_tokens": 3 } } }),
            ),
            (
                "openai",
                serde_json::json!({ "usage": {
                    "prompt_tokens": 10, "completion_tokens": 1,
                    "prompt_tokens_details": { "cache_write_tokens": "4" }
                } }),
            ),
            (
                "openai",
                serde_json::json!({ "usage": {
                    "prompt_tokens": 10, "completion_tokens": 1,
                    "prompt_tokens_details": { "cache_write_tokens": u64::MAX }
                } }),
            ),
            (
                "ollama",
                serde_json::json!({ "done": false, "prompt_eval_count": 10, "eval_count": 1,
                    "prompt_eval_cached_count": 4 }),
            ),
            (
                "ollama",
                serde_json::json!({ "done": true, "prompt_eval_count": 10, "eval_count": 1,
                    "prompt_eval_cached_count": null }),
            ),
            (
                "anthropic",
                serde_json::json!({ "usage": { "input_tokens": 12, "output_tokens": 2,
                    "output_tokens_details": { "thinking_tokens": "1" } } }),
            ),
        ];
        for (provider, body) in cases {
            assert!(
                build_generation_event(provider, "m", None, &body.to_string(), "query", 10)
                    .is_none(),
                "{provider} reported an invalid usage count: {body}"
            );
        }
    }

    #[test]
    fn generation_event_none_on_missing_usage_or_bad_json() {
        assert!(
            build_generation_event("openai", "m", None, r#"{"choices":[]}"#, "query", 5).is_none()
        );
        assert!(build_generation_event("openai", "m", None, "not json", "query", 5).is_none());
    }

    /// TD-SANDHI-3 adversarial-review guard: usage that parses to all zeros is no measurement —
    /// empty objects, `null`, wrong-typed values, and explicit zeros must all emit nothing
    /// rather than a `final`+`provider_reported` zero-token event (sandhi-core's own Cohere
    /// hardening, applied at our dispatch so every provider arm gets it).
    #[test]
    fn generation_event_drops_all_zero_usage() {
        let zero_bodies = [
            r#"{"usage":{}}"#,
            r#"{"usage":null}"#,
            r#"{"usage":"x"}"#,
            r#"{"usage":[]}"#,
            r#"{"usage":{"prompt_tokens":"50","completion_tokens":"10"}}"#,
            r#"{"usage":{"prompt_tokens":50.7}}"#,
            r#"{"usage":{"prompt_tokens":-5,"completion_tokens":0}}"#,
            r#"{"usage":{"prompt_cache_hit_tokens":0}}"#,
        ];
        for body in zero_bodies {
            assert!(
                build_generation_event("openai", "m", None, body, "query", 5).is_none(),
                "body should emit no event: {body}"
            );
            assert!(
                build_generation_event("anthropic", "m", None, body, "query", 5).is_none(),
                "body should emit no event: {body}"
            );
        }
        // Ollama reports `prompt_eval_count: 0` on cached prompts — zeros drop, real counts emit.
        assert!(
            build_generation_event(
                "ollama",
                "m",
                None,
                r#"{"prompt_eval_count":0}"#,
                "query",
                5
            )
            .is_none()
        );
        let cached = build_generation_event(
            "ollama",
            "m",
            None,
            r#"{"prompt_eval_count":0,"eval_count":9}"#,
            "query",
            5,
        )
        .unwrap();
        assert_eq!(cached.tokens_out, 9);

        // A cache-only measurement is still a measurement: fresh input 0 + cache_read 40 emits.
        let cache_only = serde_json::json!({
            "usage": { "prompt_tokens": 40, "prompt_tokens_details": { "cached_tokens": 40 } }
        })
        .to_string();
        let ev = build_generation_event("openai", "m", None, &cache_only, "query", 5).unwrap();
        assert_eq!(ev.tokens_in, 0);
        assert_eq!(ev.cache_read_tokens, 40);

        // DeepSeek top-level cache arm, hit > prompt (saturating): a real cache read emits.
        let deepseek = serde_json::json!({
            "usage": { "prompt_tokens": 10, "prompt_cache_hit_tokens": 50 }
        })
        .to_string();
        let ev = build_generation_event("openai", "m", None, &deepseek, "query", 5).unwrap();
        assert_eq!(ev.tokens_in, 0);
        assert_eq!(ev.cache_read_tokens, 50);

        // Reasoning-only counts as signal too (billable unfolds reasoning when it exceeds
        // completion) — the guard must not drop it.
        let reasoning_only =
            serde_json::json!({ "usage": { "completion_tokens_details": { "reasoning_tokens": 50 } } })
                .to_string();
        let ev =
            build_generation_event("openai", "o4-mini", None, &reasoning_only, "query", 5).unwrap();
        assert_eq!(ev.reasoning_tokens, Some(50));
    }

    /// The pre-0.1.5-sync wire shape: an event that sets none of the new optional fields must
    /// serialize **byte-identically** to what this module emitted before the mirror-drift fix,
    /// so downstream log pipelines see no change until a field is actually populated.
    #[test]
    fn pre_sync_shape_serializes_byte_identically() {
        let mut ev = UsageEvent::external_embedding(
            "azure_openai",
            "text-embedding-3-small",
            Some("tenant-abc"),
            1234,
            "embed_batch",
        );
        ev.request_id = "keu-emb-0-0".to_string();
        ev.occurred_at = "2026-01-01T00:00:00+00:00".to_string();
        let json = serde_json::to_string(&ev).unwrap();
        assert_eq!(
            json,
            concat!(
                "{\"schema_version\":\"1\",",
                "\"request_id\":\"keu-emb-0-0\",",
                "\"occurred_at\":\"2026-01-01T00:00:00+00:00\",",
                "\"provider\":\"azure_openai\",",
                "\"model\":\"text-embedding-3-small\",",
                "\"backend\":\"external\",",
                "\"group_id\":\"tenant-abc\",",
                "\"route\":\"embed_batch\",",
                "\"tokens_in\":1234,",
                "\"tokens_out\":0,",
                "\"cache_creation_tokens\":0,",
                "\"cache_read_tokens\":0}"
            )
        );
    }

    /// Contract proof: whatever this mirror emits must deserialize into the **authoritative**
    /// `sandhi_core::UsageEvent` (usage-event.v1), with omitted optionals landing on the schema
    /// defaults.
    #[test]
    fn old_shape_deserializes_into_sandhi_core_event_with_defaults() {
        let ev = UsageEvent::external_embedding("openai", "m", Some("t"), 10, "embed_batch");
        let json = serde_json::to_string(&ev).unwrap();
        let wire: sandhi_core::UsageEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(wire.tokens_in, 10);
        assert_eq!(wire.backend, sandhi_core::Backend::External);
        assert_eq!(
            wire.usage_completeness,
            sandhi_core::UsageCompleteness::Unavailable
        );
        assert_eq!(wire.usage_basis, sandhi_core::UsageBasis::ProviderReported);
        assert_eq!(wire.attempts, 1);
        assert!(wire.reasoning_tokens.is_none());
        assert!(wire.duration_ms.is_none());
    }

    /// The additive usage-event.v1 fields round-trip with the schema's snake_case spellings,
    /// through the authoritative crate type and back.
    #[test]
    fn new_optional_fields_round_trip_with_schema_spelling() {
        let mut ev = UsageEvent::generation_llm(
            "openai",
            "o4-mini",
            Some("tenant-x"),
            50,
            10,
            0,
            5,
            "query",
            BACKEND_EXTERNAL,
        );
        ev.usage_completeness = Some(sandhi_core::UsageCompleteness::Final);
        ev.usage_basis = Some(sandhi_core::UsageBasis::Estimated);
        ev.attempts = Some(2);
        ev.outcome = Some("success".to_string());
        ev.upstream_request_id = Some("req_abc".to_string());
        ev.duration_ms = Some(1234);
        ev.time_to_first_token_ms = Some(56);
        ev.reasoning_tokens = Some(7);
        ev.idempotency_key = Some("idem-1".to_string());
        ev.run_id = Some("run-1".to_string());
        ev.step_id = Some("step-2".to_string());
        ev.parent_id = Some("run-0".to_string());
        ev.trace_context = Some("00-abc-def-01".to_string());
        let v = serde_json::to_value(&ev).unwrap();
        assert_eq!(v["usage_completeness"], "final");
        assert_eq!(v["usage_basis"], "estimated");
        assert_eq!(v["attempts"], 2);
        assert_eq!(v["outcome"], "success");
        assert_eq!(v["upstream_request_id"], "req_abc");
        assert_eq!(v["duration_ms"], 1234);
        assert_eq!(v["time_to_first_token_ms"], 56);
        assert_eq!(v["reasoning_tokens"], 7);
        assert_eq!(v["run_id"], "run-1");
        // Round-trip through the authoritative sandhi-core event type.
        let wire: sandhi_core::UsageEvent = serde_json::from_value(v).unwrap();
        assert_eq!(
            wire.usage_completeness,
            sandhi_core::UsageCompleteness::Final
        );
        assert_eq!(wire.usage_basis, sandhi_core::UsageBasis::Estimated);
        assert_eq!(wire.attempts, 2);
        assert_eq!(wire.reasoning_tokens, Some(7));
        assert_eq!(wire.duration_ms, Some(1234));
        assert_eq!(wire.time_to_first_token_ms, Some(56));
        assert_eq!(wire.run_id.as_deref(), Some("run-1"));
        assert_eq!(wire.step_id.as_deref(), Some("step-2"));
        assert_eq!(wire.parent_id.as_deref(), Some("run-0"));
        assert_eq!(wire.idempotency_key.as_deref(), Some("idem-1"));
        assert_eq!(wire.trace_context.as_deref(), Some("00-abc-def-01"));
        assert_eq!(wire.upstream_request_id.as_deref(), Some("req_abc"));
        assert_eq!(wire.outcome.as_deref(), Some("success"));
    }

    /// TD-SANDHI-3: the embedding path stamps measurement provenance — provider-reported vs the
    /// heuristic (estimated) count — plus the adapter-boundary wall clock and the terminal
    /// outcome, and all spellings round-trip through the authoritative sandhi-core event type.
    #[test]
    fn embedding_measurement_stamps_basis_completeness_duration() {
        let reported = UsageEvent::external_embedding(
            "azure_openai",
            "text-embedding-3-small",
            Some("t"),
            1000,
            "embed_batch",
        )
        .with_provenance(sandhi_core::UsageBasis::ProviderReported, Some(4321));
        let v = serde_json::to_value(&reported).unwrap();
        assert_eq!(v["usage_basis"], "provider_reported");
        assert_eq!(v["usage_completeness"], "final");
        assert_eq!(v["duration_ms"], 4321);
        assert_eq!(v["outcome"], "success");
        let wire: sandhi_core::UsageEvent = serde_json::from_value(v).unwrap();
        assert_eq!(wire.usage_basis, sandhi_core::UsageBasis::ProviderReported);
        assert_eq!(
            wire.usage_completeness,
            sandhi_core::UsageCompleteness::Final
        );
        assert_eq!(wire.duration_ms, Some(4321));
        assert_eq!(wire.outcome.as_deref(), Some("success"));

        // The heuristic path (provider reports no usage → count×512) is the estimated basis.
        let estimated = UsageEvent::external_embedding(
            "byo",
            "https://byo.example/v1",
            Some("t"),
            512,
            "embed_batch",
        )
        .with_provenance(sandhi_core::UsageBasis::Estimated, Some(7));
        let v2 = serde_json::to_value(&estimated).unwrap();
        assert_eq!(v2["usage_basis"], "estimated");
        assert_eq!(v2["usage_completeness"], "final");
        assert_eq!(v2["duration_ms"], 7);
        assert_eq!(v2["outcome"], "success");
        let wire2: sandhi_core::UsageEvent = serde_json::from_value(v2).unwrap();
        assert_eq!(wire2.usage_basis, sandhi_core::UsageBasis::Estimated);
        assert_eq!(
            wire2.usage_completeness,
            sandhi_core::UsageCompleteness::Final
        );
        assert_eq!(wire2.outcome.as_deref(), Some("success"));

        // A non-carrier batch-mate: duration omitted, provenance still stamped.
        let mate = UsageEvent::external_embedding(
            "byo",
            "https://byo.example/v1",
            Some("t"),
            512,
            "embed_batch",
        )
        .with_provenance(sandhi_core::UsageBasis::Estimated, None);
        let v3 = serde_json::to_value(&mate).unwrap();
        assert!(v3.get("duration_ms").is_none());
        assert_eq!(v3["usage_basis"], "estimated");
        assert_eq!(v3["usage_completeness"], "final");
        assert_eq!(v3["outcome"], "success");
    }

    /// sandhi-core 0.1.5 parsers surface `reasoning_tokens` (OpenAI o-series); the generation
    /// path threads it through — present when reported, **absent** (not zero) otherwise, so
    /// non-reasoning events keep the old shape.
    #[test]
    fn generation_event_surfaces_reasoning_tokens_when_reported() {
        let body = serde_json::json!({
            "usage": {
                "prompt_tokens": 50, "completion_tokens": 30,
                "completion_tokens_details": { "reasoning_tokens": 12 }
            }
        })
        .to_string();
        let ev = build_generation_event("openai", "o4-mini", None, &body, "query", 9).unwrap();
        assert_eq!(ev.reasoning_tokens, Some(12));
        assert_eq!(serde_json::to_value(&ev).unwrap()["reasoning_tokens"], 12);

        let plain = serde_json::json!({ "usage": { "prompt_tokens": 5, "completion_tokens": 2 } })
            .to_string();
        let ev2 = build_generation_event("openai", "gpt-4o", None, &plain, "query", 9).unwrap();
        assert!(ev2.reasoning_tokens.is_none());
        assert!(
            serde_json::to_value(&ev2)
                .unwrap()
                .get("reasoning_tokens")
                .is_none()
        );
    }

    /// 0.1.5 meter hardening flows through: a malformed cache split (`cached > prompt`) clamps
    /// fresh input to 0 instead of underflowing into a garbage meter value.
    #[test]
    fn hardened_parser_guards_flow_through() {
        let body = serde_json::json!({
            "usage": {
                "prompt_tokens": 10, "completion_tokens": 1,
                "prompt_tokens_details": { "cached_tokens": 50 }
            }
        })
        .to_string();
        let ev = build_generation_event("openai", "m", None, &body, "query", 3).unwrap();
        assert_eq!(ev.tokens_in, 0);
        assert_eq!(ev.cache_read_tokens, 50);
        assert_eq!(ev.tokens_out, 1);
    }

    #[test]
    fn emit_is_inert_without_flag() {
        // Just assert it never panics when the flag is unset (default posture).
        UsageEvent::external_embedding("openai", "text-embedding-3-large", None, 10, "embed_batch")
            .emit();
    }
}
