//! # Authentication Middleware
//!
//! JWT and token-based authentication for all protocols.

use std::sync::Arc;

/// Authentication middleware configuration
#[derive(Debug, Clone)]
pub struct AuthConfig {
    pub jwt_secret: Option<String>,
    pub enable_jwt: bool,
    pub enable_api_key: bool,
}

impl Default for AuthConfig {
    fn default() -> Self {
        Self {
            jwt_secret: None,
            enable_jwt: false,
            enable_api_key: true,
        }
    }
}

/// Authentication middleware
pub struct AuthMiddleware {
    #[allow(dead_code)]
    config: Arc<AuthConfig>,
}

impl AuthMiddleware {
    pub fn new(config: AuthConfig) -> Self {
        Self {
            config: Arc::new(config),
        }
    }
}

// TODO: Move authentication logic from src/network/middleware/auth.rs

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_auth_config_keeps_api_keys_enabled_without_jwt_secret() {
        let config = AuthConfig::default();

        assert_eq!(config.jwt_secret, None);
        assert!(!config.enable_jwt);
        assert!(config.enable_api_key);
    }

    #[test]
    fn auth_middleware_stores_config_behind_shared_pointer() {
        let middleware = AuthMiddleware::new(AuthConfig {
            jwt_secret: Some("secret".to_string()),
            enable_jwt: true,
            enable_api_key: false,
        });

        assert_eq!(middleware.config.jwt_secret.as_deref(), Some("secret"));
        assert!(middleware.config.enable_jwt);
        assert!(!middleware.config.enable_api_key);
    }
}

