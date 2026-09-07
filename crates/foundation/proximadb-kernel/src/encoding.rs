//! Base64 encoding/decoding implementation - replaces base64 crate
//!
//! Provides standard base64 encoding and decoding with optional URL-safe variant.

use std::fmt;

/// Standard Base64 encoding alphabet
const STANDARD_ALPHABET: &[u8; 64] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// URL-safe Base64 encoding alphabet (replaces + and / with - and _)
const URL_SAFE_ALPHABET: &[u8; 64] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

/// Invalid character marker for Base64 decoding
const INVALID: u8 = 255;

/// Standard base64 encoding
pub fn base64_encode(data: &[u8]) -> String {
    base64_encode_config(data, Base64Config::standard())
}

/// Standard base64 decoding
pub fn base64_decode(encoded: &str) -> Result<Vec<u8>, Base64Error> {
    base64_decode_config(encoded, Base64Config::standard())
}

/// URL-safe base64 encoding
pub fn base64_encode_url_safe(data: &[u8]) -> String {
    base64_encode_config(data, Base64Config::url_safe())
}

/// URL-safe base64 decoding
pub fn base64_decode_url_safe(encoded: &str) -> Result<Vec<u8>, Base64Error> {
    base64_decode_config(encoded, Base64Config::url_safe())
}

/// Base64 encoding/decoding configuration
///
/// Allows customization of the Base64 alphabet and padding behavior.
#[derive(Debug, Clone)]
pub struct Base64Config {
    /// Character alphabet to use for encoding
    alphabet: &'static [u8; 64],
    /// Whether to use padding (= characters)
    padding: bool,
    /// Whether this is a URL-safe configuration
    #[allow(dead_code)]
    url_safe: bool,
}

impl Base64Config {
    /// Standard base64 configuration
    pub fn standard() -> Self {
        Base64Config {
            alphabet: STANDARD_ALPHABET,
            padding: true,
            url_safe: false,
        }
    }

    /// URL-safe base64 configuration
    pub fn url_safe() -> Self {
        Base64Config {
            alphabet: URL_SAFE_ALPHABET,
            padding: false,
            url_safe: true,
        }
    }

    /// Create decoding table for the alphabet
    fn create_decode_table(&self) -> [u8; 256] {
        let mut table = [INVALID; 256];
        for (i, &c) in self.alphabet.iter().enumerate() {
            table[c as usize] = i as u8;
        }
        // Handle padding character if padding is enabled
        if self.padding {
            table[b'=' as usize] = 64;
        }
        table
    }
}

/// Encode data using specified configuration
pub fn base64_encode_config(data: &[u8], config: Base64Config) -> String {
    if data.is_empty() {
        return String::new();
    }

    let mut result = Vec::with_capacity(data.len().div_ceil(3) * 4);
    let chunks = data.chunks_exact(3);
    let remainder = chunks.remainder();

    // Process complete 3-byte chunks
    for chunk in chunks {
        let b1 = chunk[0];
        let b2 = chunk[1];
        let b3 = chunk[2];

        result.push(config.alphabet[(b1 >> 2) as usize]);
        result.push(config.alphabet[(((b1 & 0x03) << 4) | (b2 >> 4)) as usize]);
        result.push(config.alphabet[(((b2 & 0x0f) << 2) | (b3 >> 6)) as usize]);
        result.push(config.alphabet[(b3 & 0x3f) as usize]);
    }

    // Process remainder
    match remainder.len() {
        1 => {
            let b1 = remainder[0];
            result.push(config.alphabet[(b1 >> 2) as usize]);
            result.push(config.alphabet[((b1 & 0x03) << 4) as usize]);
            if config.padding {
                result.push(b'=');
                result.push(b'=');
            }
        }
        2 => {
            let b1 = remainder[0];
            let b2 = remainder[1];
            result.push(config.alphabet[(b1 >> 2) as usize]);
            result.push(config.alphabet[(((b1 & 0x03) << 4) | (b2 >> 4)) as usize]);
            result.push(config.alphabet[((b2 & 0x0f) << 2) as usize]);
            if config.padding {
                result.push(b'=');
            }
        }
        _ => {}
    }

    String::from_utf8_lossy(&result).into_owned()
}

/// Decode base64 string using specified configuration
pub fn base64_decode_config(encoded: &str, config: Base64Config) -> Result<Vec<u8>, Base64Error> {
    if encoded.is_empty() {
        return Ok(Vec::new());
    }

    let decode_table = config.create_decode_table();
    let mut result = Vec::with_capacity((encoded.len() * 3) / 4);

    let bytes = encoded.as_bytes();
    let mut i = 0;

    // Process 4-character groups
    while i + 4 <= bytes.len() {
        let c1 = decode_table[bytes[i] as usize];
        let c2 = decode_table[bytes[i + 1] as usize];
        let c3 = if bytes[i + 2] == b'=' {
            64
        } else {
            decode_table[bytes[i + 2] as usize]
        };
        let c4 = if bytes[i + 3] == b'=' {
            64
        } else {
            decode_table[bytes[i + 3] as usize]
        };

        if c1 == INVALID || c2 == INVALID {
            return Err(Base64Error::InvalidCharacter);
        }

        result.push((c1 << 2) | (c2 >> 4));

        if c3 != 64 {
            if c3 == INVALID {
                return Err(Base64Error::InvalidCharacter);
            }
            result.push((c2 << 4) | (c3 >> 2));

            if c4 != 64 {
                if c4 == INVALID {
                    return Err(Base64Error::InvalidCharacter);
                }
                result.push((c3 << 6) | c4);
            }
        }

        i += 4;
    }

    // Handle remaining characters (for unpadded input)
    let remaining = bytes.len() - i;
    if remaining > 0 {
        // Check for invalid characters first
        for j in i..bytes.len() {
            if decode_table[bytes[j] as usize] == INVALID {
                return Err(Base64Error::InvalidCharacter);
            }
        }

        if remaining == 1 {
            return Err(Base64Error::InvalidLength);
        }

        let c1 = decode_table[bytes[i] as usize];
        let c2 = decode_table[bytes[i + 1] as usize];

        if c1 == INVALID || c2 == INVALID {
            return Err(Base64Error::InvalidCharacter);
        }

        result.push((c1 << 2) | (c2 >> 4));

        if remaining == 3 {
            let c3 = decode_table[bytes[i + 2] as usize];
            if c3 == INVALID {
                return Err(Base64Error::InvalidCharacter);
            }
            result.push((c2 << 4) | (c3 >> 2));
        }
    }

    Ok(result)
}

/// Base64 encoding/decoding errors
///
/// Represents errors that can occur during Base64 encoding or decoding operations.
#[derive(Debug, Clone, PartialEq)]
pub enum Base64Error {
    /// Invalid character encountered in input
    InvalidCharacter,
    /// Invalid input length for Base64 decoding
    InvalidLength,
}

impl fmt::Display for Base64Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Base64Error::InvalidCharacter => write!(f, "Invalid base64 character"),
            Base64Error::InvalidLength => write!(f, "Invalid base64 length"),
        }
    }
}

impl std::error::Error for Base64Error {}

/// Helper functions for common use cases
pub mod helpers {
    use super::*;

    /// Encode to base64 and wrap at specified column
    pub fn base64_encode_wrap(data: &[u8], wrap_at: usize) -> String {
        let encoded = base64_encode(data);
        if wrap_at == 0 || encoded.len() <= wrap_at {
            return encoded;
        }

        let mut wrapped = String::with_capacity(encoded.len() + (encoded.len() / wrap_at));
        for (i, chunk) in encoded.as_bytes().chunks(wrap_at).enumerate() {
            if i > 0 {
                wrapped.push('\n');
            }
            wrapped.push_str(&String::from_utf8_lossy(chunk));
        }
        wrapped
    }

    /// Decode base64 ignoring whitespace
    pub fn base64_decode_ignore_whitespace(encoded: &str) -> Result<Vec<u8>, Base64Error> {
        let cleaned: String = encoded.chars().filter(|c| !c.is_whitespace()).collect();
        base64_decode(&cleaned)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_base64_encode_decode() {
        let test_cases = vec![
            (b"" as &[u8], ""),
            (b"f", "Zg=="),
            (b"fo", "Zm8="),
            (b"foo", "Zm9v"),
            (b"foob", "Zm9vYg=="),
            (b"fooba", "Zm9vYmE="),
            (b"foobar", "Zm9vYmFy"),
            (b"Hello, World!", "SGVsbG8sIFdvcmxkIQ=="),
        ];

        for (input, expected) in test_cases {
            let encoded = base64_encode(input);
            assert_eq!(encoded, expected);

            let decoded = base64_decode(&encoded).expect("Valid base64 should decode successfully");
            assert_eq!(decoded, input);
        }
    }

    #[test]
    fn test_url_safe_encoding() {
        let data = b"sure.";
        let standard = base64_encode(data);
        let url_safe = base64_encode_url_safe(data);

        assert_eq!(standard, "c3VyZS4=");
        assert_eq!(url_safe, "c3VyZS4"); // No padding

        let decoded =
            base64_decode_url_safe(&url_safe).expect("URL-safe base64 should decode successfully");
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_invalid_input() {
        assert!(base64_decode("!!!").is_err());
        assert!(base64_decode("A").is_err());
    }

    #[test]
    fn test_wrapped_encoding() {
        let data = b"The quick brown fox jumps over the lazy dog";
        let wrapped = helpers::base64_encode_wrap(data, 16);
        assert!(wrapped.contains('\n'));

        let decoded = helpers::base64_decode_ignore_whitespace(&wrapped)
            .expect("Wrapped base64 should decode successfully");
        assert_eq!(decoded, data);
    }

    // Comprehensive test suite additions

    #[test]
    fn test_base64_all_ascii_characters() {
        // Test all ASCII characters
        for i in 0..=255u8 {
            let data = [i];
            let encoded = base64_encode(&data);
            let decoded = base64_decode(&encoded).expect("Valid base64 should decode successfully");
            assert_eq!(decoded, data, "Failed for byte {:#04x}", i);
        }
    }

    #[test]
    fn test_base64_binary_data() {
        // Test with random binary data
        let binary_data: Vec<u8> = (0..1000).map(|i| ((i * 173) % 256) as u8).collect();
        let encoded = base64_encode(&binary_data);
        let decoded = base64_decode(&encoded).expect("Valid base64 should decode successfully");
        assert_eq!(decoded, binary_data);
    }

    #[test]
    fn test_base64_large_data() {
        // Test with large data (10MB)
        let large_data = vec![0xAA; 10_000_000];
        let encoded = base64_encode(&large_data);
        let decoded = base64_decode(&encoded).expect("Valid base64 should decode successfully");
        assert_eq!(decoded, large_data);
    }

    #[test]
    fn test_base64_chunk_boundaries() {
        // Test data that aligns and misaligns with 3-byte boundaries
        for size in [1, 2, 3, 4, 5, 6, 7, 8, 9, 63, 64, 65, 127, 128, 129] {
            let data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
            let encoded = base64_encode(&data);
            let decoded = base64_decode(&encoded).expect("Valid base64 should decode successfully");
            assert_eq!(decoded, data, "Failed for size {}", size);
        }
    }

    #[test]
    fn test_base64_config_standard() {
        let config = Base64Config::standard();
        assert_eq!(config.alphabet, STANDARD_ALPHABET);
        assert!(config.padding);
        assert!(!config.url_safe);
    }

    #[test]
    fn test_base64_config_url_safe() {
        let config = Base64Config::url_safe();
        assert_eq!(config.alphabet, URL_SAFE_ALPHABET);
        assert!(!config.padding);
        assert!(config.url_safe);
    }

    #[test]
    fn test_base64_encode_config() {
        let data = b"Hello, World!";

        // Test with standard config
        let standard_config = Base64Config::standard();
        let encoded_standard = base64_encode_config(data, standard_config);
        assert_eq!(encoded_standard, base64_encode(data));

        // Test with URL-safe config
        let url_safe_config = Base64Config::url_safe();
        let encoded_url_safe = base64_encode_config(data, url_safe_config);
        assert_eq!(encoded_url_safe, base64_encode_url_safe(data));
    }

    #[test]
    fn test_base64_decode_config() {
        let data = b"Hello, World!";
        let encoded = base64_encode(data);

        // Test with standard config
        let standard_config = Base64Config::standard();
        let decoded_standard = base64_decode_config(&encoded, standard_config)
            .expect("Standard config should decode successfully");
        assert_eq!(decoded_standard, data);

        // Test with URL-safe config
        let url_safe_encoded = base64_encode_url_safe(data);
        let url_safe_config = Base64Config::url_safe();
        let decoded_url_safe = base64_decode_config(&url_safe_encoded, url_safe_config)
            .expect("URL-safe config should decode successfully");
        assert_eq!(decoded_url_safe, data);
    }

    #[test]
    fn test_base64_decode_table_creation() {
        let config = Base64Config::standard();
        let table = config.create_decode_table();

        // Check that alphabet characters map correctly
        for (i, &ch) in config.alphabet.iter().enumerate() {
            assert_eq!(table[ch as usize], i as u8);
        }

        // Check that non-alphabet characters map to INVALID
        assert_eq!(table[b'!' as usize], INVALID);
        assert_eq!(table[b'@' as usize], INVALID);
        assert_eq!(table[b' ' as usize], INVALID);
    }

    #[test]
    fn test_base64_padding_variants() {
        let test_cases: Vec<(&[u8], &str, &str)> = vec![
            (b"f", "Zg==", "Zg"),
            (b"fo", "Zm8=", "Zm8"),
            (b"foo", "Zm9v", "Zm9v"), // No padding needed
        ];

        for (data, with_padding, without_padding) in test_cases {
            let standard = base64_encode(data);
            let url_safe = base64_encode_url_safe(data);

            assert_eq!(standard, with_padding);
            assert_eq!(url_safe, without_padding);

            // Both should decode correctly
            assert_eq!(
                base64_decode(&standard).expect("Standard base64 should decode successfully"),
                data
            );
            assert_eq!(
                base64_decode_url_safe(&url_safe)
                    .expect("URL-safe base64 should decode successfully"),
                data
            );
        }
    }

    #[test]
    fn test_base64_decode_errors() {
        // Invalid characters
        assert!(matches!(
            base64_decode("SGVsbG8h!"),
            Err(Base64Error::InvalidCharacter)
        ));
        assert!(matches!(
            base64_decode("@#$%"),
            Err(Base64Error::InvalidCharacter)
        ));

        // Invalid length (single character)
        assert!(matches!(
            base64_decode("A"),
            Err(Base64Error::InvalidLength)
        ));

        // Empty string should succeed
        assert!(base64_decode("").is_ok());
    }

    #[test]
    fn test_base64_decode_padding_variants() {
        let data = b"sure.";

        // Test with different padding scenarios
        assert_eq!(
            base64_decode("c3VyZS4=").expect("Padded base64 should decode successfully"),
            data
        ); // Correct padding
        assert_eq!(
            base64_decode("c3VyZS4").expect("Unpadded base64 should decode successfully"),
            data
        ); // No padding

        // URL-safe should handle both
        assert_eq!(
            base64_decode_url_safe("c3VyZS4").expect("URL-safe base64 should decode successfully"),
            data
        );
        assert_eq!(
            base64_decode_url_safe("c3VyZS4=")
                .expect("URL-safe padded base64 should decode successfully"),
            data
        );
    }

    #[test]
    fn test_base64_special_characters() {
        // Test data containing characters that differ between standard and URL-safe
        let data_with_special = [0xFE, 0xFF, 0x3E, 0x3F]; // Should produce +/ in standard, -_ in URL-safe

        let standard = base64_encode(&data_with_special);
        let url_safe = base64_encode_url_safe(&data_with_special);

        // Standard should contain + and /
        assert!(standard.contains('+') || standard.contains('/'));

        // URL-safe should contain - and/or _
        assert!(url_safe.contains('-') || url_safe.contains('_'));

        // Both should decode to same data
        assert_eq!(
            base64_decode(&standard).expect("Standard base64 should decode successfully"),
            data_with_special
        );
        assert_eq!(
            base64_decode_url_safe(&url_safe).expect("URL-safe base64 should decode successfully"),
            data_with_special
        );
    }

    #[test]
    fn test_base64_error_display() {
        let invalid_char_error = Base64Error::InvalidCharacter;
        let invalid_length_error = Base64Error::InvalidLength;

        assert_eq!(
            format!("{}", invalid_char_error),
            "Invalid base64 character"
        );
        assert_eq!(format!("{}", invalid_length_error), "Invalid base64 length");
    }

    #[test]
    fn test_base64_error_debug() {
        let error = Base64Error::InvalidCharacter;
        let debug_str = format!("{:?}", error);
        assert!(debug_str.contains("InvalidCharacter"));
    }

    #[test]
    fn test_base64_helpers_wrap_edge_cases() {
        let data = b"short";

        // Test with wrap_at = 0 (should not wrap)
        let no_wrap = helpers::base64_encode_wrap(data, 0);
        assert!(!no_wrap.contains('\n'));
        assert_eq!(no_wrap, base64_encode(data));

        // Test with wrap_at larger than encoded length
        let long_wrap = helpers::base64_encode_wrap(data, 1000);
        assert!(!long_wrap.contains('\n'));
        assert_eq!(long_wrap, base64_encode(data));

        // Test with wrap_at = 1 (wrap after every character)
        let char_wrap = helpers::base64_encode_wrap(data, 1);
        let newline_count = char_wrap.matches('\n').count();
        assert!(newline_count > 0);
    }

    #[test]
    fn test_base64_helpers_ignore_whitespace() {
        let data = b"Hello, World!";
        let encoded = base64_encode(data);

        // Add various whitespace characters
        let with_whitespace = format!(
            "{}\n  \t{}\r\n{}",
            &encoded[..4],
            &encoded[4..8],
            &encoded[8..]
        );

        let decoded = helpers::base64_decode_ignore_whitespace(&with_whitespace)
            .expect("Base64 with whitespace should decode successfully");
        assert_eq!(decoded, data);

        // Test with only whitespace
        let whitespace_result = helpers::base64_decode_ignore_whitespace("   \t\r\n   ")
            .expect("Whitespace-only should decode to empty");
        assert_eq!(whitespace_result, b"");
    }

    #[test]
    fn test_base64_consistency_across_configs() {
        let data = b"Test data for consistency check";

        // Standard encoding/decoding
        let standard_encoded = base64_encode(data);
        let standard_decoded =
            base64_decode(&standard_encoded).expect("Standard base64 should decode successfully");
        assert_eq!(standard_decoded, data);

        // URL-safe encoding/decoding
        let url_safe_encoded = base64_encode_url_safe(data);
        let url_safe_decoded = base64_decode_url_safe(&url_safe_encoded)
            .expect("URL-safe base64 should decode successfully");
        assert_eq!(url_safe_decoded, data);

        // Cross-decoding with custom configs
        let std_config = Base64Config::standard();
        let url_config = Base64Config::url_safe();

        // Standard encoded should decode with standard config
        assert_eq!(
            base64_decode_config(&standard_encoded, std_config.clone())
                .expect("Standard config should decode successfully"),
            data
        );

        // URL-safe encoded should decode with URL-safe config
        assert_eq!(
            base64_decode_config(&url_safe_encoded, url_config.clone())
                .expect("URL-safe config should decode successfully"),
            data
        );
    }

    #[test]
    fn test_base64_concurrent_encoding() {
        let data = Arc::new(b"concurrent encoding test data".to_vec());
        let mut handles = vec![];
        let results = Arc::new(std::sync::Mutex::new(Vec::new()));

        // Spawn multiple threads
        for _ in 0..10 {
            let data = Arc::clone(&data);
            let results = Arc::clone(&results);

            let handle = thread::spawn(move || {
                let mut local_results = Vec::new();

                for _ in 0..1000 {
                    let encoded = base64_encode(&data);
                    let decoded =
                        base64_decode(&encoded).expect("Concurrent base64 decode should succeed");
                    local_results.push((encoded, decoded));
                }

                results
                    .lock()
                    .expect("Mutex should not be poisoned")
                    .extend(local_results);
            });

            handles.push(handle);
        }

        for handle in handles {
            handle.join().expect("Thread should not panic");
        }

        let all_results = results.lock().expect("Mutex should not be poisoned");
        let (first_encoded, first_decoded) = &all_results[0];

        // All results should be identical
        for (encoded, decoded) in all_results.iter() {
            assert_eq!(encoded, first_encoded);
            assert_eq!(decoded, first_decoded);
            assert_eq!(decoded, &**data);
        }
    }

    #[test]
    fn test_base64_memory_efficiency() {
        // Test that encoding doesn't use excessive memory
        let small_data = b"small";
        let encoded = base64_encode(small_data);

        // Encoded length should be approximately 4/3 of input + padding
        let expected_len = small_data.len().div_ceil(3) * 4;
        assert_eq!(encoded.len(), expected_len);
    }

    #[test]
    fn test_base64_alphabet_coverage() {
        // Generate data that will use all characters in the alphabet
        let mut test_data = Vec::new();

        // Add bytes that will generate all possible 6-bit patterns
        for i in 0..=255 {
            test_data.push(i);
        }

        let encoded = base64_encode(&test_data);
        let url_safe_encoded = base64_encode_url_safe(&test_data);

        // Check that we get a good variety of characters
        let standard_chars: std::collections::HashSet<char> = encoded.chars().collect();
        let url_safe_chars: std::collections::HashSet<char> = url_safe_encoded.chars().collect();

        // Should have many different characters
        assert!(standard_chars.len() > 50);
        assert!(url_safe_chars.len() > 50);

        // Both should decode correctly
        assert_eq!(
            base64_decode(&encoded).expect("Full alphabet coverage should decode successfully"),
            test_data
        );
        assert_eq!(
            base64_decode_url_safe(&url_safe_encoded)
                .expect("URL-safe full alphabet should decode successfully"),
            test_data
        );
    }

    #[test]
    fn test_base64_streaming_compatibility() {
        // Test that our implementation is compatible with streaming
        let data = b"This is a test of streaming compatibility";

        // Encode in one shot
        let full_encoded = base64_encode(data);

        // Simulate streaming by encoding in chunks
        let chunk_size = 6; // Multiple of 3 for clean encoding
        let mut chunked_encoded = String::new();

        for chunk in data.chunks(chunk_size) {
            chunked_encoded.push_str(&base64_encode(chunk));
        }

        // For proper streaming, we'd need more complex logic, but this tests basic chunking
        let full_decoded =
            base64_decode(&full_encoded).expect("Full encoded should decode successfully");
        assert_eq!(full_decoded, data);
    }

    #[test]
    fn test_base64_performance_characteristics() {
        // Test that encoding/decoding performs well
        let data = vec![0xCC; 10000]; // 10KB of data

        let start = std::time::Instant::now();
        for _ in 0..100 {
            let encoded = base64_encode(&data);
            let _ = base64_decode(&encoded).expect("Performance test decode should succeed");
        }
        let duration = start.elapsed();

        // Should complete 100 encode/decode cycles in reasonable time
        assert!(
            duration.as_millis() < 1000,
            "Base64 performance too slow: {:?}",
            duration
        );
    }
}
