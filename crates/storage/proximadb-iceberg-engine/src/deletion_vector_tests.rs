//! The golden bytes below come from an independent implementation, CRoaring
//! (`pyroaring` 1.0.0 `BitMap64(...).run_optimize(); .serialize()`), and the
//! blob framing from Python's `zlib.crc32`. Matching them byte for byte is what
//! makes the encoding readable by other Iceberg engines.

use super::*;
use sha2::{Digest, Sha256};

fn unhex(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).expect("hex"))
        .collect()
}

fn set(positions: &[u64]) -> PositionDeletes {
    let mut d = PositionDeletes::new();
    for p in positions {
        d.insert(*p);
    }
    d
}

const SMALL: &[u64] = &[0, 5, 7, 65535, 65536, 100000];
const SMALL_HEX: &str = "0100000000000000000000003a3000000200000000000300010001001800000020000000\
                         000005000700ffff0000a086";

#[test]
fn encoding_matches_croaring_byte_for_byte() {
    let cases: &[(&[u64], &str)] = &[
        (&[], "0000000000000000"),
        (SMALL, SMALL_HEX),
        (
            &[3, (1 << 32) + 1, (1 << 32) + 70000],
            "0200000000000000000000003a3000000100000000000000100000000300010000003a3000000200\
             00000000000001000000180000001a00000001007011",
        ),
        (
            &[0, 65536, 2 * 65536, 3 * 65536, 4 * 65536],
            "0100000000000000000000003a300000050000000000000001000000020000000300000004000000\
             300000003200000034000000360000003800000000000000000000000000",
        ),
    ];
    for (positions, hex) in cases {
        let hex: String = hex.split_whitespace().collect();
        let d = set(positions);
        assert_eq!(d.encode_portable(), unhex(&hex), "{positions:?}");
        assert_eq!(
            PositionDeletes::decode_portable(&unhex(&hex))
                .expect("decode")
                .positions(),
            positions.to_vec()
        );
    }
}

#[test]
fn a_bitmap_container_matches_croaring() {
    let positions: Vec<u64> = (0..20000).step_by(2).collect();
    let bytes = set(&positions).encode_portable();
    assert_eq!(bytes.len(), 8220);
    let digest: String = Sha256::digest(&bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert_eq!(
        digest,
        "a51a126839bd2377d502f4273a9324be52be042765baa3704db7a131ebb3d0d1"
    );
    let back = PositionDeletes::decode_portable(&bytes).expect("decode");
    assert_eq!(back.positions(), positions);
}

#[test]
fn run_containers_from_other_writers_decode() {
    // Run cookie, two run containers then an array, no offset header (< 4 containers).
    let bytes = unhex(
        "0100000000000000000000003b3002000300003f1f01000200030000000100e8033f1f010070110200\
         400d",
    );
    let mut expected: Vec<u64> = (1000..9000).collect();
    expected.extend([70000, 70001, 70002, 200000]);
    let d = PositionDeletes::decode_portable(&bytes).expect("decode");
    assert_eq!(d.cardinality(), expected.len() as u64);
    assert_eq!(d.positions(), expected);
}

#[test]
fn the_blob_framing_matches_the_spec() {
    let blob = set(SMALL).encode_blob();
    let expected = format!("00000034d1d33964{SMALL_HEX}b50c9855");
    let expected: String = expected.split_whitespace().collect();
    assert_eq!(blob, unhex(&expected));
    assert_eq!(
        PositionDeletes::decode_blob(&blob)
            .expect("decode")
            .positions(),
        SMALL.to_vec()
    );
}

#[test]
fn a_damaged_blob_is_rejected() {
    let blob = set(SMALL).encode_blob();
    let mut flipped = blob.clone();
    flipped[20] ^= 1;
    assert!(PositionDeletes::decode_blob(&flipped).is_err(), "CRC");
    let mut magic = blob.clone();
    magic[4] = 0;
    assert!(PositionDeletes::decode_blob(&magic).is_err(), "magic");
    assert!(
        PositionDeletes::decode_blob(&blob[..blob.len() - 1]).is_err(),
        "length"
    );
    let mut trailing = set(SMALL).encode_portable();
    trailing.push(0);
    assert!(
        PositionDeletes::decode_portable(&trailing).is_err(),
        "trailing"
    );
}

#[test]
fn union_keeps_both_sets() {
    let mut a = set(&[1, 2, (1 << 32) + 9]);
    a.union_with(&set(&[2, 3, (2 << 32)]));
    assert_eq!(a.positions(), vec![1, 2, 3, (1 << 32) + 9, 2 << 32]);
    assert_eq!(a.cardinality(), 5);
    assert!(a.contains(3) && !a.contains(4));
}
