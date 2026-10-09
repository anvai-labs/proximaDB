//! TD-PROTO-2: conformance guard for the hand-maintained v1 Rust mirrors.
//!
//! The v1 proto packages are NOT code-generated: `build.rs` regenerates only the
//! self-contained `proximadb.v2` package, and the v1 mirrors under
//! The v1 mirrors (`crates/foundation/proximadb-proto/src/proto/*.v1.rs` + this crate's `src/ranking.rs`) are hand-maintained to protect
//! hand-written serde impls. Nothing else checks that they still match the
//! `.proto` sources — a field added to a proto but not to the mirror silently
//! decodes to `None`/default on the binary wire, and a field added only to the
//! mirror is dead code. Both directions were invisible to every gate until this
//! test (the live instance it was filed on: `CatalogConfig`'s missing
//! `oltp = 17` oneof arm).
//!
//! How it works:
//! 1. When `protoc` is available, compile `proto/proximadb/v1/*.proto` +
//!    `proto/proximadb/explain.proto` into a `FileDescriptorSet`
//!    (`--include_imports`) and decode it with `prost-types`.
//! 2. Parse the mirror sources for their `#[prost(...)]` attributes, derives,
//!    enum discriminants and `from_str_name` tables (the mirrors carry no
//!    runtime reflection, so the source is the only self-description).
//! 3. Compare per package: message sets, field names/tags/kinds/cardinality,
//!    oneof member tags, and enum value wire-name → discriminant mappings, in
//!    BOTH directions (proto→mirror catches missing fields, mirror→proto
//!    catches invented ones).
//!
//! Skips gracefully when `protoc` is absent (CI's unit-test job installs
//! `protobuf-compiler`, so the gate is exercised there). When `protoc` exists,
//! current findings are ratcheted against the checked-in ledger
//! (`v1_conformance_known_drift.json`): NEW drift fails the test, and FIXED
//! drift makes the stale ledger entry fail until the ledger is shrunk — the
//! count only goes down (same semantics as the repo's other quality ratchets).
//!
//! Note: this file lives under `src/proto/` so the v1-proto-usage migration
//! ratchet (`scripts/check_v1_proto_usage.py`) does not count it — that metric
//! measures migration progress, and this harness is contract infrastructure
//! over the mirror itself, not a new consumer of it.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::process::Command;

use prost::Message as _;
use prost_types::field_descriptor_proto::{Label, Type};
use prost_types::{DescriptorProto, FieldDescriptorProto, FileDescriptorProto, FileDescriptorSet};

/// (proto package, mirror source file relative to this crate's manifest).
const MIRRORS: &[(&str, &str)] = &[
    ("proximadb.v1", "src/proto/proximadb.v1.rs"),
    ("proximadb.cluster.v1", "src/proto/proximadb.cluster.v1.rs"),
    (
        "proximadb.streaming.v1",
        "src/proto/proximadb.streaming.v1.rs",
    ),
    ("proximadb.explain.v1", "src/proto/proximadb.explain.v1.rs"),
    ("proximadb.v1.ranking", "src/ranking.rs"),
];

/// Packages with their own drift gates — never compared here.
const EXCLUDED_PACKAGES: &[&str] = &["proximadb.v2"];

/// Checked-in ledger of KNOWN pre-existing drift (ratchet semantics, mirroring
/// `UPDATE_OPENAPI_SPEC`): the test fails on any finding NOT in the snapshot
/// (new drift) and on any snapshot entry that no longer reproduces (fixed —
/// shrink the ledger). Regenerate after fixing drift with:
///
/// ```sh
/// UPDATE_V1_DRIFT_SNAPSHOT=1 cargo test -p proximadb-proto --lib v1_conformance
/// ```
const SNAPSHOT_PATH: &str = "src/proto/v1_conformance_known_drift.json";

// ---------------------------------------------------------------------------
// Mirror source parsing
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
struct MirrorField {
    name: String,
    tag: u32,
    kind: String,
    repeated: bool,
    optional: bool,
    map: Option<(String, String)>,
    /// Referenced message/enum type, for type-ref comparison.
    type_ref: Option<String>,
}

#[derive(Debug, Default)]
struct MirrorOneofField {
    /// Struct field name (== proto oneof name).
    name: String,
    /// The `oneof = "mod::Enum"` reference, verbatim.
    r#ref: String,
    tags: Vec<u32>,
}

#[derive(Debug, Default)]
struct MirrorMessage {
    fields: Vec<MirrorField>,
    oneof_fields: Vec<MirrorOneofField>,
}

#[derive(Debug, Default)]
struct MirrorOneof {
    /// (variant name, tag, kind, payload type ref)
    variants: Vec<(String, u32, String, Option<String>)>,
}

#[derive(Debug, Default)]
struct MirrorEnum {
    /// (variant name, discriminant)
    values: Vec<(String, i32)>,
    /// wire name -> variant name, from `from_str_name`.
    from_str_name: BTreeMap<String, String>,
    /// variant name -> wire name, from `as_str_name`.
    as_str_name: BTreeMap<String, String>,
}

/// Everything the source parser indexed, plus anything it could not classify —
/// non-empty anomalies are a failure: unrecognized syntax must never silently
/// dodge the gate.
#[derive(Debug, Default)]
struct MirrorModel {
    messages: BTreeMap<String, MirrorMessage>,
    oneofs: BTreeMap<String, MirrorOneof>,
    enums: BTreeMap<String, MirrorEnum>,
    /// tonic service stubs: full service name → method name → streaming
    /// flavor (`unary` / `server_streaming` / `streaming`), from the client
    /// mod (the server mod repeats the same SERVICE_NAME).
    services: BTreeMap<String, BTreeMap<String, String>>,
    anomalies: Vec<String>,
}

enum ItemKind {
    /// A `#[prost::Message]` struct.
    Message(String),
    /// A `#[prost::Oneof]` enum.
    Oneof(String),
    /// A `#[prost::Enumeration]` enum.
    Enum(String),
}

#[derive(Default)]
struct ParsedProstAttr {
    kind: Option<String>,
    r#ref: Option<String>,
    map_kv: Option<(String, String)>,
    repeated: bool,
    optional: bool,
    tag: Option<u32>,
    tags: Vec<u32>,
}

/// Split `#[prost(...)]` content on commas outside quoted strings.
fn split_attr_tokens(body: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut cur = String::new();
    let mut in_quote = false;
    for ch in body.chars() {
        match ch {
            '"' => {
                in_quote = !in_quote;
                cur.push(ch);
            }
            ',' if !in_quote => {
                tokens.push(cur.trim().to_string());
                cur.clear();
            }
            _ => cur.push(ch),
        }
    }
    if !cur.trim().is_empty() {
        tokens.push(cur.trim().to_string());
    }
    tokens
}

/// Parse one complete `#[prost(...)]` attribute; `Err` on unrecognized tokens
/// (fail-loud).
fn parse_prost_attr(attr: &str) -> Result<ParsedProstAttr, String> {
    let body = attr
        .strip_prefix("#[prost(")
        .and_then(|b| b.strip_suffix(")]"))
        .ok_or_else(|| format!("not a prost attribute: {attr}"))?;
    let mut out = ParsedProstAttr::default();
    for token in split_attr_tokens(body) {
        if token.is_empty() {
            continue;
        }
        if let Some(value) = token.strip_prefix("tag = ") {
            out.tag = Some(
                value
                    .trim_matches('"')
                    .parse()
                    .map_err(|_| format!("unparseable tag in {attr}"))?,
            );
        } else if let Some(value) = token.strip_prefix("tags = ") {
            for digits in value.trim_matches('"').split(',') {
                let digits = digits.trim();
                if !digits.is_empty() {
                    out.tags.push(
                        digits
                            .parse()
                            .map_err(|_| format!("unparseable tags entry in {attr}"))?,
                    );
                }
            }
        } else if let Some(value) = token.strip_prefix("enumeration = ") {
            out.r#ref = Some(value.trim_matches('"').to_string());
            out.kind = Some("enumeration".into());
        } else if let Some(value) = token.strip_prefix("map = ") {
            let kv = value.trim_matches('"');
            let (k, v) = kv
                .split_once(',')
                .ok_or_else(|| format!("bad map in {attr}"))?;
            out.map_kv = Some((k.trim().to_string(), v.trim().to_string()));
            out.kind = Some("map".into());
        } else if let Some(value) = token.strip_prefix("oneof = ") {
            out.r#ref = Some(value.trim_matches('"').to_string());
            out.kind = Some("oneof".into());
        } else if let Some((kind, hint)) = token.split_once(" = ") {
            // prost collection hints: `bytes = "vec"`, `message = "btreemap"`, …
            // — they select the Rust container, not the wire shape.
            let hint = hint.trim_matches('"');
            if !matches!(hint, "vec" | "btreemap" | "hashmap") {
                return Err(format!("unrecognized prost attr token {token:?} in {attr}"));
            }
            out.kind = Some(kind.to_string());
        } else if token == "optional" {
            out.optional = true;
        } else if token == "repeated" {
            out.repeated = true;
        } else if token == "packed" {
            // Recognized and ignored: packing only affects the wire layout of
            // repeated scalars, which tag equality already guards.
        } else if !token.contains('=') && !token.contains('(') {
            out.kind = Some(token);
        } else {
            return Err(format!("unrecognized prost attr token {token:?} in {attr}"));
        }
    }
    if out.kind.is_none() {
        return Err(format!("prost attribute without a kind: {attr}"));
    }
    Ok(out)
}

/// heck-style CamelCase → snake_case (splits before the last uppercase of an
/// acronym run: "SqlValue" → "sql_value", "OLTPConfig" → "oltp_config").
fn snake(name: &str) -> String {
    let chars: Vec<char> = name.chars().collect();
    let mut out = String::new();
    for (i, &ch) in chars.iter().enumerate() {
        let prev_lower = i > 0 && chars[i - 1].is_lowercase();
        let next_lower = chars.get(i + 1).is_some_and(|c| c.is_lowercase());
        if ch.is_uppercase() && (prev_lower || next_lower) && !out.is_empty() {
            out.push('_');
        }
        out.push(ch.to_ascii_lowercase());
    }
    out
}

/// snake_case → PascalCase ("string_value" → "StringValue").
fn pascal(name: &str) -> String {
    name.split('_')
        .map(|part| {
            let mut c = part.chars();
            match c.next() {
                Some(first) => first.to_ascii_uppercase().to_string() + c.as_str(),
                None => String::new(),
            }
        })
        .collect()
}

/// Last Rust identifier in a field or oneof payload type. Generated mirror
/// shapes wrap message types in paths and containers such as
/// `Option<super::Foo>` / `Vec<crate::Bar>`; the final identifier is the
/// protobuf type name that must agree with the descriptor.
fn final_rust_type_ident(ty: &str) -> Option<String> {
    ty.split(|ch: char| !ch.is_alphanumeric() && ch != '_')
        .rfind(|part| !part.is_empty())
        .map(str::to_string)
}

/// Normalize a type path key: snake_case the LAST component only ("A::B::Camel"
/// → "A::B::camel"). Message/enum names are compared in snake space so a
/// hand-maintainer's casing rename (RbacAuditEvent vs proto's RBACAuditEvent)
/// is not flagged — the wire contract carries tags, not Rust type names.
fn normalize_type_path(path: &str) -> String {
    match path.rsplit_once("::") {
        Some((head, last)) => format!("{head}::{}", snake(last)),
        None => snake(path),
    }
}

/// Depth delta of a line, with string-literal contents and trailing `//`
/// comments masked — braces inside them (e.g. `write!(f, "{{")`) must not
/// desync the counter for the rest of the file.
fn brace_delta(line: &str) -> isize {
    let mut cleaned = String::with_capacity(line.len());
    let mut in_string = false;
    let mut escaped = false;
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        if in_string {
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_string = false;
            }
            continue;
        }
        match c {
            '"' => {
                in_string = true;
                cleaned.push(' ');
            }
            '/' if chars.peek() == Some(&'/') => break,
            _ => cleaned.push(c),
        }
    }
    cleaned.matches('{').count() as isize - cleaned.matches('}').count() as isize
}

/// Whether a `#[derive(...)]` line includes a prost trait under the given
/// bare or path-qualified name (`Message`, `::prost::Message`, …).
fn derive_has(derive: &str, trait_name: &str) -> bool {
    let inner = derive
        .trim_start_matches("#[derive(")
        .trim_end_matches(")]");
    inner.split(',').any(|token| {
        let token = token.trim();
        token == trait_name || token.ends_with(&format!("::{trait_name}"))
    })
}

/// `pub struct Foo …{` / `pub enum Foo {` / `pub mod foo {` → item name.
fn opener_name(line: &str, prefix: &str) -> Option<String> {
    let rest = line.strip_prefix(prefix)?;
    if !rest.ends_with('{') {
        return None;
    }
    let name = rest
        .trim_end_matches('{')
        .trim()
        .split(['<', ' ', '('])
        .next()
        .unwrap_or_default()
        .trim()
        .to_string();
    if name.is_empty() || !name.chars().all(|c| c.is_alphanumeric() || c == '_') {
        return None;
    }
    Some(name)
}

/// `pub struct Foo {}` — a one-line empty struct (prost renders empty proto
/// messages this way). Returns the item name.
fn unit_struct_name(line: &str) -> Option<String> {
    if !line.starts_with("pub struct ") || !line.ends_with("{}") {
        return None;
    }
    let name = line
        .strip_prefix("pub struct ")?
        .trim_end_matches("{}")
        .trim()
        .to_string();
    if name.is_empty() || !name.chars().all(|c| c.is_alphanumeric() || c == '_') {
        return None;
    }
    Some(name)
}

fn parse_enum_value(line: &str) -> Option<(String, i32)> {
    let line = line.split("//").next().unwrap_or("").trim();
    let line = line.trim_end_matches(',');
    let (name, number) = line.split_once('=')?;
    let number: i32 = number.trim().parse().ok()?;
    let name = name.trim();
    if !name.is_empty() && name.chars().all(|c| c.is_alphanumeric() || c == '_') {
        Some((name.to_string(), number))
    } else {
        None
    }
}

fn parse_from_str_arm(line: &str) -> Option<(String, String)> {
    // "WIRE_NAME" => Some(Self::Variant), — also tolerates the qualified
    // EnumName::Variant spelling its as_str_name twin accepts (round 17: a
    // correctly-written qualified arm parsed as an empty table and tripped
    // the fail-loud guard on a correct mirror).
    if !line.contains("\" =>") {
        return None; // `_ => None,` and friends
    }
    let (wire, rest) = line.split_once("=>")?;
    let wire = wire.trim().trim_matches('"');
    let rest = rest.trim();
    let variant = rest
        .strip_prefix("Some(")
        .and_then(|inner| inner.strip_suffix("),").or_else(|| inner.strip_suffix(')')))
        .map(|path| {
            // Self::Variant or EnumName::Variant — the VARIANT is the last
            // segment either way.
            path.rsplit("::").next().unwrap_or_default()
        })
        .unwrap_or("");
    if wire.is_empty()
        || variant.is_empty()
        || variant.contains(|c: char| !c.is_alphanumeric() && c != '_')
    {
        return None;
    }
    Some((wire.to_string(), variant.to_string()))
}

fn parse_as_str_arm(line: &str) -> Option<(String, String)> {
    // Self::Variant => "WIRE_NAME",  (or `EnumName::Variant => "WIRE_NAME",`)
    if !line.contains("=> \"") {
        return None;
    }
    let (variant, wire) = line.split_once("=>")?;
    let variant = variant.trim().rsplit("::").next()?.trim();
    let wire = wire.trim().trim_end_matches(',').trim().trim_matches('"');
    if variant.is_empty()
        || wire.is_empty()
        || variant.contains(|c: char| !c.is_alphanumeric() && c != '_')
        || wire.contains(|c: char| !c.is_alphanumeric() && c != '_')
    {
        return None;
    }
    Some((variant.to_string(), wire.to_string()))
}

/// Parse one mirror source file into a [`MirrorModel`].
fn parse_mirror(source: &str, file_label: &str) -> MirrorModel {
    let mut model = MirrorModel::default();
    let module_key = |modules: &[(String, usize)]| {
        modules
            .iter()
            .map(|(n, _)| n.as_str())
            .collect::<Vec<_>>()
            .join("::")
    };

    // Parser state.
    let mut depth: usize = 0;
    // (module name, depth of its CONTENTS — opener line depth + 1).
    let mut modules: Vec<(String, usize)> = Vec::new();
    let mut attr_buf = String::new();
    let mut last_derive = String::new();
    let mut pending_prost: Option<String> = None;
    let mut pending_variant_prost: Option<String> = None;
    let mut item: Option<ItemKind> = None;
    let mut item_depth: usize = 0;
    // Open `impl NAME {` block: (enum key it feeds, opener line depth).
    let mut open_impl: Option<(String, usize)> = None;
    let mut in_from_str_name = false;
    let mut in_as_str_name = false;
    // tonic service method buffering: module path → (`pub async fn` name,
    // streaming flavor).
    let mut mod_async_fns: BTreeMap<String, BTreeMap<String, String>> = BTreeMap::new();

    let source_lines: Vec<String> = source.lines().map(str::to_string).collect();
    for (line_index, raw) in source_lines.iter().enumerate() {
        let raw = raw.as_str();
        let line = raw.trim();

        // ---- attributes (accumulate until brackets balance) ----
        if !attr_buf.is_empty() || line.starts_with("#[") {
            attr_buf.push_str(line);
            let opens = attr_buf.matches('[').count();
            let closes = attr_buf.matches(']').count();
            if opens > 0 && closes >= opens {
                let attr = std::mem::take(&mut attr_buf);
                if attr.starts_with("#[derive(") {
                    last_derive = attr;
                } else if attr.starts_with("#[prost(") {
                    // Inside a oneof enum body the attr belongs to the next
                    // variant line; elsewhere to the next field line.
                    if matches!(item, Some(ItemKind::Oneof(_))) {
                        pending_variant_prost = Some(attr);
                    } else if pending_prost.is_some() {
                        model
                            .anomalies
                            .push(format!("{file_label}: two prost attrs in a row: {attr}"));
                    } else {
                        pending_prost = Some(attr);
                    }
                }
                // serde/doc/repr/other attrs: ignored.
            }
            continue; // attributes never contribute braces
        }

        // ---- comment-only lines ----
        if line.is_empty() || line.starts_with("//") {
            continue;
        }

        // ---- tonic service stubs: SERVICE_NAME consts + client methods ----
        // tonic emits the methods FIRST and the SERVICE_NAME const at the
        // BOTTOM of the client mod, so buffer pub async fns per module and
        // merge them when the const identifies the service.
        if let Some(rest) = line.strip_prefix("pub async fn ")
            && let Some((name, _)) = rest.split_once('(')
        {
            let name = name.trim();
            if !name.is_empty() && name.chars().all(|c| c.is_alphanumeric() || c == '_') {
                // Streaming flavor: scan this client method through the next
                // method boundary for its tonic dispatch call. Normalize
                // client-streaming and bidi to the descriptor comparison's
                // shared `streaming` bucket.
                let flavor = source_lines
                    .iter()
                    .skip(line_index + 1)
                    .take_while(|line| {
                        !line.trim().starts_with("pub async fn ")
                            && !line.trim().starts_with("pub const SERVICE_NAME")
                    })
                    .find_map(|l| {
                        [
                            ("unary", "unary"),
                            ("server_streaming", "server_streaming"),
                            ("client_streaming", "streaming"),
                            ("streaming", "streaming"),
                        ]
                        .iter()
                        .find_map(|(call, flavor)| {
                            l.contains(&format!("self.inner.{call}"))
                                .then(|| flavor.to_string())
                        })
                    })
                    .unwrap_or_default();
                mod_async_fns
                    .entry(module_key(&modules))
                    .or_default()
                    .insert(name.to_string(), flavor);
            }
            continue;
        }
        if let Some(rest) = line.strip_prefix("pub const SERVICE_NAME")
            && let Some(full) = rest.split('"').nth(1)
        {
            // tonic layout: the rpc methods are `pub async fn`s in the
            // `<service>_client` module while the SERVICE_NAME const sits at
            // the bottom of `<service>_server` — merge from either home.
            let mk = module_key(&modules);
            let client_mk = mk.strip_suffix("_server").map(|s| format!("{s}_client"));
            let methods = mod_async_fns
                .remove(&mk)
                .or_else(|| client_mk.and_then(|c| mod_async_fns.remove(&c)))
                .unwrap_or_default();
            model
                .services
                .entry(full.to_string())
                .or_default()
                .extend(methods);
            continue;
        }

        // ---- impl blocks feed the enum wire-name tables ----
        if line.starts_with("impl ") && line.ends_with('{') {
            let name = line
                .strip_prefix("impl ")
                .and_then(|r| r.strip_suffix(" {"))
                .unwrap_or_default()
                .trim()
                .to_string();
            // Keys normalize the type name with snake(), matching how enum
            // items are keyed (hand-maintainers rename e.g. RBACAuditEvent).
            let name = normalize_type_path(&name);
            let key = if name.contains("::") {
                name
            } else {
                let mk = module_key(&modules);
                if mk.is_empty() {
                    name
                } else {
                    format!("{mk}::{name}")
                }
            };
            open_impl = Some((key, depth));
            in_from_str_name = false;
        } else if let Some((key, impl_depth)) = &open_impl {
            if depth == *impl_depth + 1 && line.starts_with("pub fn ") {
                in_from_str_name = line.starts_with("pub fn from_str_name");
                in_as_str_name = line.starts_with("pub fn as_str_name");
            } else if depth >= *impl_depth + 2 {
                if in_from_str_name {
                    if let Some((wire, variant)) = parse_from_str_arm(line) {
                        model
                            .enums
                            .entry(key.clone())
                            .or_default()
                            .from_str_name
                            .insert(wire, variant);
                    }
                } else if in_as_str_name {
                    // Self::Variant => "WIRE",  (also `EnumName::Variant => …`)
                    if let Some((variant, wire)) = parse_as_str_arm(line) {
                        model
                            .enums
                            .entry(key.clone())
                            .or_default()
                            .as_str_name
                            .insert(variant, wire);
                    }
                }
            }
        }

        // ---- item bodies ----
        match &item {
            Some(ItemKind::Message(key)) => {
                if let Some(rest) = line.strip_prefix("pub ")
                    && let Some((name, ty)) = rest.split_once(':')
                {
                    // Raw identifiers (`pub r#type:`) mirror proto field
                    // names that are Rust keywords — normalize.
                    let name = name.trim().trim_start_matches("r#").to_string();
                    match pending_prost.take() {
                        None => model.anomalies.push(format!(
                            "{file_label}: {key}: pub field {name:?} has no #[prost] attribute"
                        )),
                        Some(attr) => match parse_prost_attr(&attr) {
                            Ok(parsed) => {
                                let message = model.messages.entry(key.clone()).or_default();
                                if parsed.kind.as_deref() == Some("oneof") {
                                    message.oneof_fields.push(MirrorOneofField {
                                        name,
                                        r#ref: parsed.r#ref.unwrap_or_default(),
                                        tags: parsed.tags,
                                    });
                                } else {
                                    let kind = parsed.kind.unwrap_or_default();
                                    let type_ref = match kind.as_str() {
                                        "enumeration" => {
                                            parsed.r#ref.as_deref().and_then(final_rust_type_ident)
                                        }
                                        "message" => final_rust_type_ident(ty),
                                        _ => None,
                                    };
                                    message.fields.push(MirrorField {
                                        name,
                                        tag: parsed.tag.unwrap_or(0),
                                        kind,
                                        repeated: parsed.repeated,
                                        optional: parsed.optional,
                                        map: parsed.map_kv,
                                        type_ref,
                                    });
                                }
                            }
                            Err(reason) => model.anomalies.push(format!("{file_label}: {reason}")),
                        },
                    }
                    continue; // field lines carry no braces
                }
            }
            Some(ItemKind::Oneof(key)) => {
                if let Some(attr) = pending_variant_prost.take() {
                    let variant = line
                        .split(|c: char| c == '(' || c == ',' || c.is_whitespace())
                        .next()
                        .unwrap_or_default()
                        .to_string();
                    // Payload between the parens is the variant's type — for
                    // message/enum variants this is the reference compared
                    // against the descriptor's type_name (last segment).
                    // Scalar payloads (f64, String, Vec<u8>, …) are NOT
                    // references and carry None.
                    let payload_ref = |kind: &str| {
                        if !matches!(kind, "message" | "enumeration") {
                            return None;
                        }
                        line.split_once('(')
                            .and_then(|(_, rest)| rest.rsplit_once(')'))
                            .map(|(inner, _)| inner)
                            .map(|inner| {
                                inner
                                    .rsplit("::")
                                    .next()
                                    .unwrap_or_default()
                                    .trim()
                                    .to_string()
                            })
                    };
                    match parse_prost_attr(&attr) {
                        Ok(parsed) => {
                            let kind = parsed.kind.clone().unwrap_or_default();
                            // Enumeration variants carry their reference in
                            // the attr (`enumeration = "…"`); message variants
                            // in the payload type; scalars carry none.
                            let r = match kind.as_str() {
                                "enumeration" => parsed.r#ref.as_ref().map(|rf| {
                                    rf.rsplit("::").next().unwrap_or_default().to_string()
                                }),
                                _ => payload_ref(&kind),
                            };
                            model.oneofs.entry(key.clone()).or_default().variants.push((
                                variant,
                                parsed.tag.unwrap_or(0),
                                kind,
                                r,
                            ));
                        }
                        Err(reason) => model.anomalies.push(format!("{file_label}: {reason}")),
                    }
                    continue; // variant lines carry no braces
                }
                // An identifier followed by `(` is a variant line that arrived
                // without a prost attr — a drift signal. Continuation lines of
                // a wrapped payload start with `::` and are ignored.
                let looks_like_variant = line.split_once('(').is_some_and(|(head, _)| {
                    !head.is_empty() && head.chars().all(|c| c.is_alphanumeric() || c == '_')
                });
                if looks_like_variant {
                    model.anomalies.push(format!(
                        "{file_label}: {key}: oneof variant {line:?} has no #[prost] attribute"
                    ));
                }
            }
            Some(ItemKind::Enum(key)) => {
                if let Some((name, number)) = parse_enum_value(line) {
                    model
                        .enums
                        .entry(key.clone())
                        .or_default()
                        .values
                        .push((name, number));
                }
                // Other lines (rare cfg attrs / impl headers) are handled above.
            }
            None => {}
        }

        // ---- item/module openers ----
        if let Some(name) = unit_struct_name(line)
            && derive_has(&last_derive, "Message")
        {
            // One-line empty struct: register it, no body follows.
            let mk = module_key(&modules);
            let key = if mk.is_empty() {
                name.clone()
            } else {
                format!("{mk}::{name}")
            };
            model.messages.entry(normalize_type_path(&key)).or_default();
            last_derive.clear();
        } else if let Some(name) = opener_name(line, "pub struct ") {
            item = if derive_has(&last_derive, "Message") {
                let mk = module_key(&modules);
                let key = normalize_type_path(&if mk.is_empty() {
                    name.clone()
                } else {
                    format!("{mk}::{name}")
                });
                model.messages.entry(key.clone()).or_default();
                item_depth = depth;
                Some(ItemKind::Message(key))
            } else {
                None // tonic client/server structs and friends
            };
            last_derive.clear();
        } else if let Some(name) = opener_name(line, "pub enum ") {
            let mk = module_key(&modules);
            let key = normalize_type_path(&if mk.is_empty() {
                name.clone()
            } else {
                format!("{mk}::{name}")
            });
            item = if derive_has(&last_derive, "Oneof") {
                model.oneofs.entry(key.clone()).or_default();
                item_depth = depth;
                Some(ItemKind::Oneof(key))
            } else if derive_has(&last_derive, "Enumeration") {
                model.enums.entry(key.clone()).or_default();
                item_depth = depth;
                Some(ItemKind::Enum(key))
            } else {
                None
            };
            last_derive.clear();
        } else if let Some(name) = opener_name(line, "pub mod ") {
            modules.push((name, depth + 1));
            last_derive.clear();
        }

        // ---- depth bookkeeping + closings ----
        let delta = brace_delta(line);
        if delta < 0 {
            let new_depth = depth.saturating_sub((-delta) as usize);
            if new_depth <= item_depth {
                item = None;
            }
            if let Some((_, impl_depth)) = &open_impl
                && new_depth <= *impl_depth
            {
                open_impl = None;
                in_from_str_name = false;
                in_as_str_name = false;
            }
            while modules.last().is_some_and(|(_, d)| *d > new_depth) {
                modules.pop();
            }
            depth = new_depth;
        } else {
            depth += delta as usize;
        }
    }

    model
}

// ---------------------------------------------------------------------------
// Descriptor model
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct DescField {
    name: String,
    number: u32,
    kind: &'static str,
    repeated: bool,
    proto3_optional: bool,
    is_map: bool,
    /// (key kind, value kind) from the synthetic map-entry message.
    map: Option<(String, String)>,
    /// Last segment of the descriptor's fully-qualified type_name, for
    /// message/enum type-reference comparison.
    type_ref: Option<String>,
    oneof: Option<String>,
}

impl DescField {
    /// Wire kind as it appears in a prost attribute.
    fn attr_kind(&self) -> &str {
        if self.is_map { "map" } else { self.kind }
    }
}

#[derive(Debug, Default)]
struct DescMessage {
    /// ALL fields, oneof members included.
    fields: Vec<DescField>,
    /// Declared oneof names, synthetic proto3-optional singles removed.
    oneofs: Vec<String>,
}

#[derive(Debug, Default)]
struct DescIndex {
    messages: BTreeMap<String, DescMessage>,
    /// (wire value name, number) per enum.
    enums: BTreeMap<String, Vec<(String, i32)>>,
    /// Declared services: full name → (method name, streaming flavor).
    services: BTreeMap<String, Vec<(String, String)>>,
}

fn kind_of(ty: &Type) -> Option<&'static str> {
    Some(match ty {
        Type::Double => "double",
        Type::Float => "float",
        Type::Int64 => "int64",
        Type::Uint64 => "uint64",
        Type::Int32 => "int32",
        Type::Fixed64 => "fixed64",
        Type::Fixed32 => "fixed32",
        Type::Bool => "bool",
        Type::String => "string",
        Type::Message => "message",
        Type::Bytes => "bytes",
        Type::Uint32 => "uint32",
        Type::Enum => "enumeration",
        Type::Sfixed32 => "sfixed32",
        Type::Sfixed64 => "sfixed64",
        Type::Sint32 => "sint32",
        Type::Sint64 => "sint64",
        Type::Group => return None,
    })
}

fn enum_values(en: &prost_types::EnumDescriptorProto) -> Vec<(String, i32)> {
    en.value
        .iter()
        .map(|v| (v.name.clone().unwrap_or_default(), v.number.unwrap_or(0)))
        .collect()
}

/// Map-entry messages by fully-qualified type name → (key kind, value kind).
type MapEntries = BTreeMap<String, (String, String)>;

fn build_desc_index(files: &[FileDescriptorProto]) -> DescIndex {
    let mut index = DescIndex::default();
    let mut map_entries: MapEntries = BTreeMap::new();

    /// Key/value kinds of a synthetic map-entry message (fields 1 and 2).
    fn entry_kinds(entry: &DescriptorProto) -> Option<(String, String)> {
        let kind_at = |number: i32| {
            entry
                .field
                .iter()
                .find(|f| f.number == Some(number))
                .and_then(|f| f.r#type)
                .and_then(|v| Type::try_from(v).ok())
                .and_then(|t| kind_of(&t))
                .map(str::to_string)
        };
        Some((kind_at(1)?, kind_at(2)?))
    }

    fn collect_entries(
        msg: &DescriptorProto,
        package: &str,
        ancestors: &[String],
        out: &mut MapEntries,
    ) {
        let full = if ancestors.is_empty() {
            msg.name.clone().unwrap_or_default()
        } else {
            format!(
                "{}.{}",
                ancestors.join("."),
                msg.name.clone().unwrap_or_default()
            )
        };
        for nested in &msg.nested_type {
            let is_map_entry = nested
                .options
                .as_ref()
                .is_some_and(|o| o.map_entry.unwrap_or(false));
            if is_map_entry {
                if let Some(kinds) = entry_kinds(nested) {
                    out.insert(
                        // protoc emits fully-qualified (dot-prefixed) type names.
                        format!(
                            ".{package}.{full}.{}",
                            nested.name.clone().unwrap_or_default()
                        ),
                        kinds,
                    );
                }
            } else {
                let mut chain = ancestors.to_vec();
                chain.push(msg.name.clone().unwrap_or_default());
                collect_entries(nested, package, &chain, out);
            }
        }
    }

    fn walk(
        msg: &DescriptorProto,
        ancestors: &[String],
        map_entries: &MapEntries,
        index: &mut DescIndex,
    ) {
        let chain: Vec<String> = ancestors.iter().map(|a| snake(a)).collect();
        let join = |name: &str| {
            normalize_type_path(&if chain.is_empty() {
                name.to_string()
            } else {
                format!("{}::{}", chain.join("::"), name)
            })
        };
        let key = join(msg.name.as_deref().unwrap_or_default());

        let mut out = DescMessage::default();
        let oneof_names: Vec<String> = msg
            .oneof_decl
            .iter()
            .map(|o| o.name.clone().unwrap_or_default())
            .collect();

        for field in &msg.field {
            if let Some(f) = describe_field(field, &oneof_names, map_entries) {
                out.fields.push(f);
            }
        }
        // Keep only oneofs with at least one non-synthetic member (each proto3
        // `optional` field gets a synthetic single-member oneof).
        out.oneofs = oneof_names
            .into_iter()
            .filter(|oname| {
                msg.field.iter().any(|f| {
                    !f.proto3_optional.unwrap_or(false)
                        && f.oneof_index
                            .and_then(|i| {
                                msg.oneof_decl
                                    .get(i as usize)
                                    .and_then(|o| o.name.as_deref())
                                    .map(|n| n == oname.as_str())
                            })
                            .unwrap_or(false)
                })
            })
            .collect();

        index.messages.insert(key, out);

        for nested in &msg.nested_type {
            let is_map_entry = nested
                .options
                .as_ref()
                .is_some_and(|o| o.map_entry.unwrap_or(false));
            if !is_map_entry {
                let mut chain = ancestors.to_vec();
                chain.push(msg.name.clone().unwrap_or_default());
                walk(nested, &chain, map_entries, index);
            }
        }
        for en in &msg.enum_type {
            // Nested enums are keyed under the parent message's module too:
            // `message Foo { enum Bar {} }` → "foo::bar".
            let mut echain = chain.clone();
            echain.push(snake(msg.name.as_deref().unwrap_or_default()));
            let ekey = normalize_type_path(&format!(
                "{}::{}",
                echain.join("::"),
                en.name.as_deref().unwrap_or_default()
            ));
            index.enums.insert(ekey, enum_values(en));
        }
    }

    // Two passes: map-entry kinds must be known before fields are described.
    for file in files {
        let package = file.package.clone().unwrap_or_default();
        for msg in &file.message_type {
            collect_entries(msg, &package, &[], &mut map_entries);
        }
    }
    for file in files {
        let package = file.package.clone().unwrap_or_default();
        for svc in &file.service {
            let name = format!("{package}.{}", svc.name.clone().unwrap_or_default());
            let methods = svc
                .method
                .iter()
                .filter_map(|m| {
                    let name = m.name.clone()?;
                    // Client-streaming and bidi both render `.streaming` on
                    // the tonic client side, so they share one flavor bucket.
                    let flavor = match (
                        m.client_streaming.unwrap_or(false),
                        m.server_streaming.unwrap_or(false),
                    ) {
                        (false, false) => "unary",
                        (false, true) => "server_streaming",
                        (true, _) => "streaming",
                    };
                    Some((name, flavor.to_string()))
                })
                .collect();
            index.services.insert(name, methods);
        }
        for en in &file.enum_type {
            index.enums.insert(
                snake(en.name.as_deref().unwrap_or_default()),
                enum_values(en),
            );
        }
        for msg in &file.message_type {
            walk(msg, &[], &map_entries, &mut index);
        }
    }
    index
}

fn describe_field(
    field: &FieldDescriptorProto,
    oneof_names: &[String],
    map_entries: &MapEntries,
) -> Option<DescField> {
    let ty: Type = field.r#type.and_then(|v| Type::try_from(v).ok())?;
    let label: Label = field.label.and_then(|v| Label::try_from(v).ok())?;
    let kind = kind_of(&ty)?;
    let name = field.name.clone().unwrap_or_default();

    // Map fields: protoc gives each map field a synthetic nested entry message
    // (options.map_entry) referenced by fully-qualified type_name.
    let is_map = ty == Type::Message
        && label == Label::Repeated
        && field
            .type_name
            .as_deref()
            .is_some_and(|tn| map_entries.contains_key(tn));
    let map = field
        .type_name
        .as_deref()
        .and_then(|tn| map_entries.get(tn))
        .cloned()
        .filter(|_| is_map);

    let oneof = field
        .oneof_index
        .and_then(|i| oneof_names.get(i as usize).cloned())
        .filter(|_| !field.proto3_optional.unwrap_or(false));

    Some(DescField {
        name,
        number: field.number.unwrap_or(0).unsigned_abs(),
        kind,
        repeated: label == Label::Repeated && !is_map,
        proto3_optional: field.proto3_optional.unwrap_or(false),
        is_map,
        map,
        type_ref: field
            .type_name
            .as_deref()
            .map(|tn| tn.rsplit('.').next().unwrap_or_default().to_string()),
        oneof,
    })
}

// ---------------------------------------------------------------------------
// Comparison
// ---------------------------------------------------------------------------

/// Resolve a mirror `oneof = "…"` reference against the parsed oneof enums.
/// The reference is module-RELATIVE to the field's enclosing module (e.g.
/// `evolution_change::Change` inside `schema_evolution_request`), so try the
/// message's enclosing module chain from innermost to outermost, then bare.
fn resolve_oneof_ref<'a>(
    r#ref: &str,
    message_key: &str,
    oneofs: &'a BTreeMap<String, MirrorOneof>,
) -> Option<(String, &'a MirrorOneof)> {
    let normalized = normalize_type_path(r#ref);
    let msg_comps: Vec<&str> = message_key.split("::").collect();
    let mut prefixes: Vec<String> = Vec::new();
    for take in (0..msg_comps.len().saturating_sub(1)).rev() {
        prefixes.push(msg_comps[..=take].join("::"));
    }
    prefixes.push(String::new());
    for prefix in prefixes {
        let candidate = if prefix.is_empty() {
            normalized.clone()
        } else {
            format!("{prefix}::{normalized}")
        };
        if let Some(oneof) = oneofs.get(&candidate) {
            return Some((candidate, oneof));
        }
    }
    None
}

struct Comparer {
    file_label: &'static str,
    errors: Vec<String>,
}

impl Comparer {
    fn record(&mut self, key: &str, what: String) {
        self.errors
            .push(format!("{} {key}: {what}", self.file_label));
    }

    fn compare(&mut self, desc: &DescIndex, mirror: &MirrorModel) {
        self.errors.extend(mirror.anomalies.iter().cloned());

        self.compare_message_sets(desc, mirror);
        for (key, dmsg) in &desc.messages {
            if let Some(mmsg) = mirror.messages.get(key) {
                self.compare_message(key, dmsg, mmsg, mirror);
            }
        }
        self.compare_enums(desc, mirror);
        self.compare_services(desc, mirror);
    }

    /// gRPC service stubs are consumer-facing surface: a deleted or
    /// never-written stub (or a proto rpc with no mirror method, or a stub
    /// with the wrong streaming flavor) must fail the gate, not surface as a
    /// downstream compile break.
    fn compare_services(&mut self, desc: &DescIndex, mirror: &MirrorModel) {
        for (name, dmethods) in &desc.services {
            // tonic renders rpc `AppendEntries` as fn `append_entries`.
            let dset: BTreeSet<String> = dmethods.iter().map(|(m, _)| snake(m)).collect();
            match mirror.services.get(name) {
                None => self.record(
                    "(services)",
                    format!(
                        "proto service {name:?} (methods {dmethods:?}) has no tonic stub in the mirror"
                    ),
                ),
                Some(mmethods) => {
                    let mset: BTreeSet<&String> = mmethods.keys().collect();
                    let mset_owned: BTreeSet<String> =
                        mset.iter().map(|s| (*s).clone()).collect();
                    for missing in dset.difference(&mset_owned) {
                        self.record(
                            "(services)",
                            format!("service {name:?}: rpc {missing:?} missing from mirror stub"),
                        );
                    }
                    for invented in mset_owned.difference(&dset) {
                        self.record(
                            "(services)",
                            format!(
                                "service {name:?}: mirror method {invented:?} not declared in proto"
                            ),
                        );
                    }
                    // Streaming flavor: a hand-splice that renders a
                    // streaming rpc as unary truncates/hangs the call.
                    for (dmethod, dflavor) in dmethods {
                        if let Some(mflavor) = mmethods.get(&snake(dmethod))
                            && mflavor != dflavor
                        {
                            self.record(
                                "(services)",
                                format!(
                                    "service {name:?}: rpc {dmethod:?} streaming flavor mismatch mirror {mflavor:?} vs proto {dflavor:?}"
                                ),
                            );
                        }
                    }
                }
            }
        }
        for name in mirror.services.keys() {
            if !desc.services.contains_key(name) {
                self.record(
                    "(services)",
                    format!("mirror service stub {name:?} not declared in any proto"),
                );
            }
        }
    }

    fn compare_message_sets(&mut self, desc: &DescIndex, mirror: &MirrorModel) {
        let desc_keys: BTreeSet<&String> = desc.messages.keys().collect();
        let mirror_keys: BTreeSet<&String> = mirror.messages.keys().collect();
        for missing in desc_keys.difference(&mirror_keys) {
            self.record(
                "(message set)",
                format!("proto message {missing:?} missing from mirror"),
            );
        }
        for invented in mirror_keys.difference(&desc_keys) {
            self.record(
                "(message set)",
                format!("mirror message {invented:?} not declared in any proto"),
            );
        }
    }

    fn compare_message(
        &mut self,
        key: &str,
        dmsg: &DescMessage,
        mmsg: &MirrorMessage,
        mirror: &MirrorModel,
    ) {
        let dfields: BTreeMap<&String, &DescField> = dmsg
            .fields
            .iter()
            .filter(|f| f.oneof.is_none())
            .map(|f| (&f.name, f))
            .collect();
        let mfields: BTreeMap<&String, &MirrorField> =
            mmsg.fields.iter().map(|f| (&f.name, f)).collect();

        for (name, df) in &dfields {
            match mfields.get(*name) {
                None => self.record(
                    key,
                    format!(
                        "proto field {name:?} (tag {}) missing from mirror",
                        df.number
                    ),
                ),
                Some(mf) => {
                    if mf.tag != df.number {
                        self.record(
                            key,
                            format!(
                                "field {name:?}: tag mismatch mirror {} vs proto {}",
                                mf.tag, df.number
                            ),
                        );
                    }
                    if mf.kind != df.attr_kind() {
                        self.record(
                            key,
                            format!(
                                "field {name:?}: kind mismatch mirror {:?} vs proto {:?}",
                                mf.kind,
                                df.attr_kind()
                            ),
                        );
                    } else if !df.is_map && matches!(df.kind, "message" | "enumeration") {
                        let mirror_ref = mf.type_ref.as_deref().map(snake);
                        let descriptor_ref = df.type_ref.as_deref().map(snake);
                        if mirror_ref != descriptor_ref {
                            // A field pointed at the WRONG existing message or
                            // enum passes tag+wire-kind equality. Compare the
                            // normalized final type segment too; package paths
                            // and Rust acronym casing are not wire-significant.
                            self.record(
                                key,
                                format!(
                                    "field {name:?}: {} type-ref mismatch mirror {:?} vs proto {:?}",
                                    df.kind, mf.type_ref, df.type_ref
                                ),
                            );
                        }
                    } else if let (Some((dk, dv)), Some((mk, mv))) = (&df.map, &mf.map) {
                        // Both key and value kinds must match the descriptor's
                        // synthetic entry message.
                        if mk != dk || mv != dv {
                            self.record(
                                key,
                                format!(
                                    "map field {name:?}: kinds mirror ({mk}, {mv}) vs proto ({dk}, {dv})"
                                ),
                            );
                        }
                    }
                    if mf.repeated != df.repeated {
                        self.record(
                            key,
                            format!(
                                "field {name:?}: repeated mismatch mirror {} vs proto {}",
                                mf.repeated, df.repeated
                            ),
                        );
                    }
                    // proto3 `optional` (explicit presence) on non-message
                    // fields must be mirrored with the `optional` attr. Message
                    // presence is Option-typed either way, so it is not
                    // compared.
                    if df.kind != "message" && !df.is_map && mf.optional != df.proto3_optional {
                        self.record(
                            key,
                            format!(
                                "field {name:?}: optional mismatch mirror {} vs proto {}",
                                mf.optional, df.proto3_optional
                            ),
                        );
                    }
                }
            }
        }
        // Members of descriptor oneofs the mirror models as plain fields
        // (wire-compatible flattening, accepted in compare_oneofs) must not be
        // flagged again here as invented fields.
        let flattened_members: BTreeSet<&String> = dmsg
            .oneofs
            .iter()
            .filter(|dname| !mmsg.oneof_fields.iter().any(|o| &o.name == *dname))
            .flat_map(|dname| {
                dmsg.fields
                    .iter()
                    .filter(move |f| f.oneof.as_deref() == Some(dname.as_str()))
                    .map(|f| &f.name)
            })
            .collect();
        for name in mfields.keys() {
            if !dfields.contains_key(*name) && !flattened_members.contains(*name) {
                self.record(key, format!("mirror field {name:?} not declared in proto"));
            }
        }

        self.compare_oneofs(key, dmsg, mmsg, mirror);
    }

    fn compare_oneofs(
        &mut self,
        key: &str,
        dmsg: &DescMessage,
        mmsg: &MirrorMessage,
        mirror: &MirrorModel,
    ) {
        let moneofs: BTreeMap<&String, &MirrorOneofField> =
            mmsg.oneof_fields.iter().map(|o| (&o.name, o)).collect();
        let doneofs: BTreeSet<&String> = dmsg.oneofs.iter().collect();

        for dname in &doneofs {
            let members: Vec<&DescField> = dmsg
                .fields
                .iter()
                .filter(|f| f.oneof.as_deref() == Some(dname.as_str()))
                .collect();
            match moneofs.get(*dname) {
                None => {
                    // Wire-compatible flattening: the mirror may model a proto
                    // oneof as plain optional fields with the same tags (same
                    // wire shape; loses only the one-at-a-time invariant).
                    // Accept iff EVERY member matches a plain field.
                    let flattened_ok = members.iter().all(|df| {
                        mmsg.fields.iter().any(|mf| {
                            mf.name == df.name
                                && mf.tag == df.number
                                && mf.kind == df.attr_kind()
                                && !mf.repeated
                        })
                    });
                    if !flattened_ok {
                        let names: Vec<&str> = members.iter().map(|f| f.name.as_str()).collect();
                        self.record(
                            key,
                            format!(
                                "proto oneof {dname:?} (members {names:?}) missing from mirror"
                            ),
                        );
                    }
                }
                Some(mo) => {
                    let oneof_enum = match resolve_oneof_ref(&mo.r#ref, key, &mirror.oneofs) {
                        Some((_, oe)) => oe,
                        None => {
                            self.record(
                                key,
                                format!(
                                    "oneof {dname:?}: mirror references enum {:?} which was not parsed",
                                    mo.r#ref
                                ),
                            );
                            continue;
                        }
                    };
                    let member_tags: BTreeSet<u32> = members.iter().map(|f| f.number).collect();
                    let wrapper_tags: BTreeSet<u32> = mo.tags.iter().copied().collect();
                    if member_tags != wrapper_tags {
                        self.record(
                            key,
                            format!(
                                "oneof {dname:?}: wrapper tags {wrapper_tags:?} != member tags {member_tags:?}"
                            ),
                        );
                    }
                    // 4-tuple adds the type REFERENCE for message/enum
                    // members (None for scalars) so a variant retargeted to
                    // the wrong struct/enum cannot pass on tag+kind alone.
                    let proto: BTreeSet<(String, u32, String, Option<String>)> = members
                        .iter()
                        .map(|f| {
                            let r = match f.attr_kind() {
                                "message" | "enumeration" => f.type_ref.clone(),
                                _ => None,
                            };
                            (pascal(&f.name), f.number, f.attr_kind().to_string(), r)
                        })
                        .collect();
                    let mirror_variants: BTreeSet<(String, u32, String, Option<String>)> =
                        oneof_enum.variants.iter().map(Clone::clone).collect();
                    if proto != mirror_variants {
                        let proto_dbg: Vec<String> = proto
                            .iter()
                            .map(|(n, t, k, r)| {
                                format!("{n}:{t}:{k}{}", r.as_deref().unwrap_or(""))
                            })
                            .collect();
                        let mirror_dbg: Vec<String> = mirror_variants
                            .iter()
                            .map(|(n, t, k, r)| {
                                format!("{n}:{t}:{k}{}", r.as_deref().unwrap_or(""))
                            })
                            .collect();
                        self.record(
                            key,
                            format!(
                                "oneof {dname:?}: member/variant mismatch proto [{}] vs mirror [{}]",
                                proto_dbg.join(", "),
                                mirror_dbg.join(", ")
                            ),
                        );
                    }
                }
            }
        }
        for mname in moneofs.keys() {
            if !doneofs.contains(*mname) {
                self.record(key, format!("mirror oneof {mname:?} not declared in proto"));
            }
        }
    }

    fn compare_enums(&mut self, desc: &DescIndex, mirror: &MirrorModel) {
        for (key, dvalues) in &desc.enums {
            let menum = match mirror.enums.get(key) {
                Some(e) => e,
                None => {
                    self.record(
                        "(enum set)",
                        format!("proto enum {key:?} missing from mirror"),
                    );
                    continue;
                }
            };
            let disc: BTreeMap<&String, i32> = menum.values.iter().map(|(n, v)| (n, *v)).collect();
            // Each name table is validated INDEPENDENTLY (merging them would
            // let a correct from_str_name mask a corrupted as_str_name — the
            // table live surfaces consume for serialization).
            let mut as_table: BTreeMap<&str, i32> = BTreeMap::new();
            for (variant, wire) in &menum.as_str_name {
                if let Some(&d) = disc.get(variant) {
                    as_table.insert(wire.as_str(), d);
                } else {
                    self.record(
                        key,
                        format!("as_str_name arm for unknown variant {variant:?}"),
                    );
                }
            }
            let mut from_table: BTreeMap<&str, i32> = BTreeMap::new();
            for (wire, variant) in &menum.from_str_name {
                if let Some(&d) = disc.get(variant) {
                    from_table.insert(wire.as_str(), d);
                } else {
                    self.record(
                        key,
                        format!("from_str_name arm for unknown variant {variant:?}"),
                    );
                }
            }
            for (wire, as_d) in &as_table {
                if let Some(&from_d) = from_table.get(wire)
                    && from_d != *as_d
                {
                    self.record(
                        key,
                        format!("enum value {wire:?}: as_str_name says {as_d} but from_str_name says {from_d}"),
                    );
                }
            }

            // prost::Enumeration ALWAYS generates BOTH name tables: an empty
            // table (joint or single) is drift — a deleted half-impl or a
            // scanner miss — and the weak fallbacks would let a wire-name
            // rename (the Polar/Polaris class) or a broken decode direction
            // pass green (round-8 finding: one-sided absence slipped through).
            if as_table.is_empty() || from_table.is_empty() {
                // Fail loud naming the missing table(s).
                self.record(
                    key,
                    format!(
                        "enum name table(s) missing: as_str_name {}, from_str_name {} — \
                         prost always generates both; absence is drift, not a comparable \
                         shape",
                        if as_table.is_empty() { "MISSING" } else { "ok" },
                        if from_table.is_empty() {
                            "MISSING"
                        } else {
                            "ok"
                        },
                    ),
                );
                continue;
            }
            for (wire, number) in dvalues {
                // Each table must cover the value independently (absence of a
                // whole table already failed loud above).
                let checks: [(&BTreeMap<&str, i32>, &str); 2] =
                    [(&from_table, "from_str_name"), (&as_table, "as_str_name")];
                for (table, table_name) in checks {
                    match table.get(wire.as_str()) {
                        None => self.record(
                            key,
                            format!(
                                "enum value {wire:?} = {number} missing from mirror {table_name}"
                            ),
                        ),
                        Some(&d) if d == *number => {}
                        Some(&d) => self.record(
                            key,
                            format!(
                                "enum value {wire:?}: {table_name} discriminant {d} != proto {number}"
                            ),
                        ),
                    }
                }
            }
            // Both tables key by WIRE NAME (built from the parsed arms), so
            // .keys() iterates wire names — surplus/typo arms are caught here.
            for wire in as_table.keys().chain(from_table.keys()) {
                if !dvalues.iter().any(|(w, _)| w == wire) {
                    self.record(
                        key,
                        format!("mirror enum wire name {wire:?} not declared in proto"),
                    );
                }
            }
        }
        let dkeys: BTreeSet<&String> = desc.enums.keys().collect();
        for mkey in mirror.enums.keys() {
            if !dkeys.contains(mkey) {
                self.record(
                    "(enum set)",
                    format!("mirror enum {mkey:?} not declared in any proto"),
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// protoc driver + entry point
// ---------------------------------------------------------------------------

/// The disarm gate: any skip cause (missing protoc, absent repo root,
/// unreadable proto tree) degrades to SKIP only on developer machines. Under `REQUIRE_PROTOC=1` (set by CI's unit-test job) the same
/// condition FAILS, so the gate can never silently disarm where it is the
/// only drift check — a skip and a pass stay distinguishable.
fn skip_or_fail(cause: &str) -> Option<Vec<u8>> {
    // See the doc comment above for the disarm causes and contract.
    if std::env::var_os("REQUIRE_PROTOC").is_some() {
        panic!(
            "TD-PROTO-2: the conformance gate is REQUIRED here (REQUIRE_PROTOC=1) \
             but cannot run: {cause} — refusing to silently disarm."
        );
    }
    eprintln!("SKIP: {cause} — TD-PROTO-2 mirror conformance check did not run.");
    None
}

fn compile_descriptor_set() -> Option<Vec<u8>> {
    // Probe protoc BEFORE touching the repository tree so the documented
    // graceful skip holds even where proto/ is absent (e.g. a published crate
    // running its tests without the repo checkout).
    let protoc = std::env::var_os("PROTOC").unwrap_or_else(|| "protoc".into());
    match Command::new(&protoc).arg("--version").output() {
        Ok(_) => {}
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            return skip_or_fail("protoc not found — install protobuf-compiler (or set PROTOC)");
        }
        Err(err) => panic!("failed to spawn protoc: {err}"),
    }

    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let Some(repo_root) = manifest.ancestors().nth(3) else {
        return skip_or_fail("repository root not found above the crate manifest");
    };
    let repo_root = repo_root.to_path_buf();
    let proto_root = repo_root.join("proto");

    let mut inputs = vec![proto_root.join("proximadb/explain.proto")];
    let v1_dir = proto_root.join("proximadb/v1");
    let Ok(v1_entries) = std::fs::read_dir(&v1_dir) else {
        // proto/ absent: skip on dev machines, FAIL under REQUIRE_PROTOC —
        // a repo restructure must not silently disarm the gate (round-7).
        return skip_or_fail("proto tree unreadable (moved or missing)");
    };
    let mut v1_files: Vec<PathBuf> = v1_entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "proto"))
        .collect();
    v1_files.sort();
    inputs.extend(v1_files);

    let out_dir = std::env::temp_dir().join(format!("proximadb-td-proto-2-{}", std::process::id()));
    std::fs::create_dir_all(&out_dir).expect("temp dir for descriptor set");
    let desc_path = out_dir.join("descriptor.bin");
    let _ = std::fs::remove_file(&desc_path);

    let output = match Command::new(&protoc)
        .arg("--include_imports")
        .arg("--descriptor_set_out")
        .arg(&desc_path)
        .arg("-I")
        .arg(&proto_root)
        .args(&inputs)
        .output()
    {
        Ok(output) => output,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            return skip_or_fail("protoc not found — install protobuf-compiler (or set PROTOC)");
        }
        Err(err) => panic!("failed to spawn protoc: {err}"),
    };
    if !output.status.success() {
        panic!(
            "protoc failed:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    Some(std::fs::read(&desc_path).expect("read descriptor set"))
}

#[test]
fn v1_mirrors_conform_to_proto_descriptors() {
    let Some(bytes) = compile_descriptor_set() else {
        return;
    };
    let fds = FileDescriptorSet::decode(bytes.as_slice()).expect("decode FileDescriptorSet");

    let mut by_package: BTreeMap<String, Vec<FileDescriptorProto>> = BTreeMap::new();
    for file in fds.file {
        let package = file.package.clone().unwrap_or_default();
        if package.starts_with("google.") {
            continue;
        }
        by_package.entry(package).or_default().push(file);
    }

    let mut findings: Vec<String> = Vec::new();

    for package in by_package.keys() {
        if EXCLUDED_PACKAGES.contains(&package.as_str()) {
            continue;
        }
        if !MIRRORS.iter().any(|(pkg, _)| pkg == package) {
            findings.push(format!(
                "(packages) : package {package:?} has proto sources but no mirror entry in \
                 MIRRORS — add one (or an EXCLUDED_PACKAGES entry with its authority)"
            ));
        }
    }

    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for (package, mirror_file) in MIRRORS {
        let files = match by_package.get(*package) {
            Some(files) if !files.is_empty() => files,
            _ => {
                findings.push(format!(
                    "(packages) : package {package:?}: no proto sources compiled"
                ));
                continue;
            }
        };
        let mirror_path = manifest.join(mirror_file);
        let source = std::fs::read_to_string(&mirror_path)
            .unwrap_or_else(|_| panic!("read mirror source {}", mirror_path.display()));
        let label: &'static str = mirror_file.rsplit('/').next().unwrap_or(mirror_file);

        let mirror = parse_mirror(&source, label);
        let desc = build_desc_index(files);

        let mut comparer = Comparer {
            file_label: label,
            errors: Vec::new(),
        };
        comparer.compare(&desc, &mirror);
        findings.extend(comparer.errors);
    }

    // Ratchet against the checked-in known-drift ledger.
    findings.sort();
    let snapshot_path = manifest.join(SNAPSHOT_PATH);
    if std::env::var_os("UPDATE_V1_DRIFT_SNAPSHOT").is_some() {
        let json = serde_json::json!({
            "_doc": "TD-PROTO-2 known-drift ledger: pre-existing v1-mirror ↔ proto \
                     mismatches, ratcheted DOWN only. Regenerate with \
                     UPDATE_V1_DRIFT_SNAPSHOT=1 cargo test -p proximadb-proto --lib \
                     v1_conformance after fixing drift. New drift is never added here — \
                     it is fixed.",
            "findings": findings,
        });
        let pretty = serde_json::to_string_pretty(&json).expect("serialize snapshot");
        std::fs::write(&snapshot_path, pretty + "\n").expect("write snapshot");
        eprintln!(
            "wrote {} ({} findings)",
            snapshot_path.display(),
            findings.len()
        );
        return;
    }

    let known: BTreeSet<String> = serde_json::from_str::<serde_json::Value>(
        &std::fs::read_to_string(&snapshot_path).unwrap_or_else(|_| {
            panic!(
                "read {} — run with UPDATE_V1_DRIFT_SNAPSHOT=1 to create",
                SNAPSHOT_PATH
            )
        }),
    )
    .expect("parse snapshot JSON")
    .get("findings")
    .and_then(|f| f.as_array())
    .expect("snapshot JSON has a findings array")
    .iter()
    .filter_map(|v| v.as_str().map(String::from))
    .collect();

    let current: BTreeSet<String> = findings.iter().cloned().collect();
    let new_drift: Vec<&String> = current.difference(&known).collect();
    let stale: Vec<&String> = known.difference(&current).collect();

    if !new_drift.is_empty() || !stale.is_empty() {
        let mut report = String::new();
        if !new_drift.is_empty() {
            report.push_str(&format!(
                "\nNEW drift ({} — fix it, do NOT add it to the ledger):\n  - ",
                new_drift.len()
            ));
            report.push_str(
                &new_drift
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>()
                    .join("\n  - "),
            );
        }
        if !stale.is_empty() {
            report.push_str(&format!(
                "\nSTALE ledger entries ({} — fixed or changed; regenerate with \
                 UPDATE_V1_DRIFT_SNAPSHOT=1):\n  - ",
                stale.len()
            ));
            report.push_str(
                &stale
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>()
                    .join("\n  - "),
            );
        }
        panic!("TD-PROTO-2 v1-mirror drift ratchet:{report}");
    }
}

#[test]
fn message_field_type_reference_mismatch_is_reported() {
    let mirror = parse_mirror(
        r#"
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct Holder {
    #[prost(message, optional, tag = "1")]
    pub item: ::core::option::Option<WrongMessage>,
}
"#,
        "synthetic.rs",
    );
    let mut desc = DescIndex::default();
    desc.messages.insert(
        "holder".to_string(),
        DescMessage {
            fields: vec![DescField {
                name: "item".to_string(),
                number: 1,
                kind: "message",
                repeated: false,
                proto3_optional: false,
                is_map: false,
                map: None,
                type_ref: Some("RightMessage".to_string()),
                oneof: None,
            }],
            oneofs: Vec::new(),
        },
    );

    let mut comparer = Comparer {
        file_label: "synthetic.rs",
        errors: Vec::new(),
    };
    comparer.compare(&desc, &mirror);

    assert!(
        comparer
            .errors
            .iter()
            .any(|error| error.contains("message type-ref mismatch")),
        "ordinary message-field retargeting must fail the guard: {:?}",
        comparer.errors
    );
}
