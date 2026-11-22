//! TD-SPINE-1 (ADR-098 D1.1–D1.2): the persisted-catalog-record envelope.
//!
//! Every byte the system catalog persists — WAL delta payloads, snapshot
//! images (whole-image, root and per-account shard), and the identity
//! high-water sidecar — is framed as
//!
//! ```text
//! [magic "PXCATREC":8][format_version:u8][body_codec:u8][body]
//! ```
//!
//! following the PMHEAD binary framing pattern (`manifest/head.rs`) and
//! ADR-063 D4 r3.1's plaintext-envelope shape: a fixed magic the reader can
//! sniff in one prefix compare, an explicit envelope format version, and an
//! explicit body codec. The body stays field-named MessagePack (ADR-098
//! D1.2); the envelope exists so a *future* body change (protobuf, layout
//! bump) is a non-flag-day switch for readers that already sniff.
//!
//! **Mixed-read-safe by construction (byte level):** legacy records — bare
//! field-named MessagePack written before the envelope existed — never begin
//! with the magic (a MessagePack map, which every legacy record is, starts
//! with a map marker `0x8n`/`0xde`/`0xdf`, never ASCII `P`), so the decoder
//! sniffs unconditionally: magic present ⇒ envelope (dispatch on
//! version/codec, **fail closed** on anything this binary does not
//! understand), magic absent ⇒ legacy MessagePack, readable forever.
//!
//! Per the pre-GA owner decision (2026-11-20) the *writer* ships the envelope
//! unconditionally — there is no write gate. The break is accepted and
//! documented in TD-SPINE-1: a pre-envelope binary cannot decode enveloped
//! records; a post-envelope binary decodes both directions.

use anyhow::{Result, bail};

/// Envelope magic (8 bytes). ASCII "PXCATREC" — ProximaDB catalog record.
/// Chosen so no MessagePack-encoded struct/enum (always a map or array at
/// byte 0) can ever begin with it.
pub const CATALOG_RECORD_MAGIC: &[u8; 8] = b"PXCATREC";

/// Current envelope format version. Bumping this is a new envelope layout
/// and requires teaching old readers OR accepting a documented pre-GA break.
pub const ENVELOPE_FORMAT_VERSION: u8 = 1;

/// Body codec 0: field-named MessagePack (`rmp_serde::to_vec_named`) — the
/// only codec today (ADR-098 D1.2 defers protobuf behind this field).
pub const BODY_CODEC_MSGPACK_NAMED: u8 = 0;

/// `[magic:8][format_version:1][body_codec:1]` — the fixed header length.
pub const ENVELOPE_HEADER_LEN: usize = CATALOG_RECORD_MAGIC.len() + 2;

/// The sniffed framing of a persisted catalog record.
#[derive(Debug, PartialEq, Eq)]
pub enum Enveloped<'a> {
    /// A legacy record: bare field-named MessagePack, no envelope. Readable
    /// forever (TD-SPINE-1 acceptance; CLAUDE.md #8 mixed-read).
    Legacy(&'a [u8]),
    /// An enveloped record with the header fields the caller dispatched on.
    Enveloped {
        format_version: u8,
        body_codec: u8,
        body: &'a [u8],
    },
}

/// Frame `body` with the current envelope (format version 1, MessagePack
/// named-field codec).
pub fn encode_enveloped(body: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(ENVELOPE_HEADER_LEN + body.len());
    out.extend_from_slice(CATALOG_RECORD_MAGIC);
    out.push(ENVELOPE_FORMAT_VERSION);
    out.push(BODY_CODEC_MSGPACK_NAMED);
    out.extend_from_slice(body);
    out
}

/// Serialize `value` as field-named MessagePack and frame it with the
/// envelope — the single writer-side entry point every persisted catalog
/// record goes through.
pub fn to_enveloped_vec_named<T: serde::Serialize>(value: &T) -> Result<Vec<u8>> {
    let body = serde_bridge::to_vec_named(value)?;
    Ok(encode_enveloped(&body))
}

/// Sniff and split a persisted catalog record. Magic absent ⇒ legacy
/// MessagePack. Magic present ⇒ the header must be complete and understood:
/// an unknown `format_version` or `body_codec` is a **typed, fail-closed
/// error naming the offending value** — never a guess, never a panic.
pub fn sniff_enveloped(bytes: &[u8]) -> Result<Enveloped<'_>> {
    if !bytes.starts_with(CATALOG_RECORD_MAGIC) {
        return Ok(Enveloped::Legacy(bytes));
    }
    if bytes.len() < ENVELOPE_HEADER_LEN {
        bail!(
            "catalog record envelope is truncated: {} bytes after the magic \
             (header needs {ENVELOPE_HEADER_LEN})",
            bytes.len()
        );
    }
    let format_version = bytes[CATALOG_RECORD_MAGIC.len()];
    let body_codec = bytes[CATALOG_RECORD_MAGIC.len() + 1];
    if format_version != ENVELOPE_FORMAT_VERSION {
        bail!(
            "catalog record envelope format version {format_version} is not \
             supported (this binary understands version \
             {ENVELOPE_FORMAT_VERSION}); failing closed"
        );
    }
    if body_codec != BODY_CODEC_MSGPACK_NAMED {
        bail!(
            "catalog record body codec {body_codec} is not supported (this \
             binary understands codec {BODY_CODEC_MSGPACK_NAMED} = \
             field-named MessagePack); failing closed"
        );
    }
    Ok(Enveloped::Enveloped {
        format_version,
        body_codec,
        body: &bytes[ENVELOPE_HEADER_LEN..],
    })
}

/// Decode a persisted catalog record: sniff the framing, then decode the
/// body (envelope or legacy) as field-named MessagePack.
pub fn from_enveloped_slice<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> Result<T> {
    match sniff_enveloped(bytes)? {
        Enveloped::Legacy(body) => serde_bridge::from_slice(body),
        Enveloped::Enveloped { body, .. } => serde_bridge::from_slice(body),
    }
}

/// Tiny indirection so the envelope module owns its codec choice in one
/// place and the rmp dependency never leaks into call sites.
mod serde_bridge {
    use anyhow::{Context, Result};

    pub(super) fn to_vec_named<T: serde::Serialize>(value: &T) -> Result<Vec<u8>> {
        rmp_serde::to_vec_named(value).context("encoding catalog record body")
    }

    pub(super) fn from_slice<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> Result<T> {
        rmp_serde::from_slice(bytes).context("decoding catalog record body")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq, serde::Serialize, serde::Deserialize)]
    struct Probe {
        name: String,
        seq: u64,
    }

    fn probe() -> Probe {
        Probe {
            name: "orders".into(),
            seq: 42,
        }
    }

    /// The writer frames every record with the magic + current version +
    /// MessagePack codec, and the reader round-trips it.
    #[test]
    fn envelope_round_trips() {
        let bytes = to_enveloped_vec_named(&probe()).expect("encode");
        assert!(
            bytes.starts_with(CATALOG_RECORD_MAGIC),
            "enveloped records must lead with the magic"
        );
        assert_eq!(bytes[CATALOG_RECORD_MAGIC.len()], ENVELOPE_FORMAT_VERSION);
        assert_eq!(bytes[CATALOG_RECORD_MAGIC.len() + 1], BODY_CODEC_MSGPACK_NAMED);
        let sniffed = sniff_enveloped(&bytes).expect("sniff");
        match &sniffed {
            Enveloped::Enveloped {
                format_version,
                body_codec,
                ..
            } => {
                assert_eq!(*format_version, ENVELOPE_FORMAT_VERSION);
                assert_eq!(*body_codec, BODY_CODEC_MSGPACK_NAMED);
            }
            other => panic!("expected an enveloped record, got {other:?}"),
        }
        assert_eq!(from_enveloped_slice::<Probe>(&bytes).unwrap(), probe());
    }

    /// Mixed-read: a legacy record (bare field-named MessagePack — which
    /// always begins with a map marker, never the ASCII magic) sniffs as
    /// `Legacy` and decodes unchanged.
    #[test]
    fn legacy_bare_messagepack_sniffs_as_legacy_and_decodes() {
        let legacy = rmp_serde::to_vec_named(&probe()).expect("legacy encode");
        assert!(!legacy.starts_with(CATALOG_RECORD_MAGIC));
        // A named-field MessagePack struct is a map at byte 0 — this is what
        // makes magic-prefix sniffing unambiguous.
        assert!(
            legacy[0] & 0xF0 == 0x80 || legacy[0] == 0xDE || legacy[0] == 0xDF,
            "legacy bodies must be MessagePack maps"
        );
        assert_eq!(
            sniff_enveloped(&legacy).unwrap(),
            Enveloped::Legacy(&legacy[..])
        );
        assert_eq!(from_enveloped_slice::<Probe>(&legacy).unwrap(), probe());
    }

    /// Fail-closed: an unknown envelope format version is a typed error that
    /// names the version — never a silent legacy fallback (which would
    /// mis-decode the body) and never a panic.
    #[test]
    fn unknown_format_version_fails_closed_naming_the_version() {
        let mut bytes = to_enveloped_vec_named(&probe()).unwrap();
        bytes[CATALOG_RECORD_MAGIC.len()] = ENVELOPE_FORMAT_VERSION + 1;
        let err = sniff_enveloped(&bytes).expect_err("must fail closed");
        let msg = format!("{err:#}");
        assert!(msg.contains("format version 2"), "error names the version: {msg}");
        assert!(from_enveloped_slice::<Probe>(&bytes).is_err());
    }

    /// Fail-closed: an unknown body codec (a future protobuf body read by
    /// this binary) is a typed error naming the codec.
    #[test]
    fn unknown_body_codec_fails_closed_naming_the_codec() {
        let mut bytes = to_enveloped_vec_named(&probe()).unwrap();
        bytes[CATALOG_RECORD_MAGIC.len() + 1] = BODY_CODEC_MSGPACK_NAMED + 7;
        let err = sniff_enveloped(&bytes).expect_err("must fail closed");
        let msg = format!("{err:#}");
        assert!(msg.contains("codec 7"), "error names the codec: {msg}");
    }

    /// Fail-closed: a torn envelope (magic written, header cut) is a typed
    /// truncation error with the existing torn-tail semantics — a short
    /// NON-magic buffer stays a legacy body for rmp to judge.
    #[test]
    fn truncated_envelope_header_fails_closed_but_short_legacy_is_left_to_rmp() {
        let mut torn = Vec::new();
        torn.extend_from_slice(CATALOG_RECORD_MAGIC);
        torn.push(ENVELOPE_FORMAT_VERSION);
        let err = sniff_enveloped(&torn).expect_err("must fail closed");
        assert!(format!("{err:#}").contains("truncated"));
        // A short buffer without the magic is not the envelope's business.
        assert_eq!(
            sniff_enveloped(b"\x80").unwrap(),
            Enveloped::Legacy(b"\x80")
        );
    }

    /// An empty record is a legacy body (an empty WAL payload would fail in
    /// rmp with its own decode error, not an envelope error).
    #[test]
    fn empty_buffer_sniffs_as_legacy() {
        assert_eq!(sniff_enveloped(&[]).unwrap(), Enveloped::Legacy(&[]));
    }
}
