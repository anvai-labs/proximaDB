//! Shared ranking test fixtures, enabled by the non-default `test-support` feature.

use crate::{AttributeAccess, DocHandle};
use std::collections::HashMap;

/// Map-backed attributes keyed by the exact document ID and field name.
///
/// Missing pairs return `None`; stored values are copied without normalizing
/// their floating-point bits. Enable this fixture only in dev-dependencies.
#[derive(Debug, Default)]
pub struct MapAttributeAccess {
    pub values: HashMap<(u32, String), f32>,
}

impl AttributeAccess for MapAttributeAccess {
    fn read_f32(&self, doc: DocHandle, field: &str) -> Option<f32> {
        self.values.get(&(doc.0, field.to_string())).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_distinguish_documents_and_exact_field_names() {
        let attrs = MapAttributeAccess {
            values: HashMap::from([
                ((0, "score".into()), 1.0),
                ((u32::MAX, "score".into()), 2.0),
                ((0, "Score".into()), 3.0),
                ((0, "".into()), 4.0),
            ]),
        };
        let port: &dyn AttributeAccess = &attrs;
        assert_eq!(port.read_f32(DocHandle(0), "score"), Some(1.0));
        assert_eq!(port.read_f32(DocHandle(u32::MAX), "score"), Some(2.0));
        assert_eq!(port.read_f32(DocHandle(0), "Score"), Some(3.0));
        assert_eq!(port.read_f32(DocHandle(0), ""), Some(4.0));
        assert_eq!(port.read_f32(DocHandle(1), "score"), None);
        assert_eq!(port.read_f32(DocHandle(0), "missing"), None);
        assert_eq!(port.read_f32(DocHandle(u32::MAX), "Score"), None);
        assert_eq!(
            MapAttributeAccess::default().read_f32(DocHandle(0), "score"),
            None
        );
    }

    #[test]
    fn reads_preserve_float_bits() {
        for bits in [
            0x0000_0000, // positive zero
            0x8000_0000, // negative zero
            0x0000_0001, // subnormal
            0xc020_0000, // negative finite value
            0x7f80_0000, // positive infinity
            0xff80_0000, // negative infinity
            0x7fc0_0042, // NaN payload
        ] {
            let attrs = MapAttributeAccess {
                values: HashMap::from([((7, "value".into()), f32::from_bits(bits))]),
            };
            let port: &dyn AttributeAccess = &attrs;
            assert_eq!(
                port.read_f32(DocHandle(7), "value").map(f32::to_bits),
                Some(bits)
            );
        }
    }
}
