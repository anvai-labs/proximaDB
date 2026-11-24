//! Engine-specific optimization profiles for block encoding and quantization.

/// Storage engine optimization profile
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EngineProfile {
    #[default]
    SST,
    Swift,
    Helix,
    Raptor,
    Nova,
}
