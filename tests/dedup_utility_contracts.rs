// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! Behavioral controls for the health predicate shared by three middleware layers.

use axum::{
    Router,
    body::Body,
    extract::ConnectInfo,
    http::{Request, StatusCode},
    middleware::from_fn_with_state,
};
use proximadb::network::middleware::{
    auth::{MiddlewareAuthConfig, UserInfo, auth_middleware},
    tls::{TlsClientCertConfig, TlsClientCertState, tls_client_cert_middleware},
};
use proximadb_network_middleware::rate_limit::{
    MiddlewareRateLimitConfig, RateLimitState, rate_limit_middleware,
};
use std::{collections::HashMap, sync::Arc, time::Duration};
use tower::ServiceExt;

// The shared contract deliberately uses a case-sensitive /health prefix.
const PATHS: &[(&str, bool)] = &[
    ("/health", true),
    ("/health/ready", true),
    ("/health/live", true),
    ("/healthz", true),
    ("/api/health", false),
    ("/Health", false),
    ("/collections", false),
];

fn request(path: &str) -> Request<Body> {
    Request::builder().uri(path).body(Body::empty()).unwrap()
}

#[tokio::test]
async fn auth_health_exemption_respects_configuration_and_path() {
    for require_auth_for_health in [false, true] {
        let config = MiddlewareAuthConfig {
            enabled: true,
            require_auth_for_health,
            api_keys: HashMap::new(),
        };
        let app = Router::new()
            .fallback(|| async { StatusCode::NO_CONTENT })
            .layer(from_fn_with_state(config, auth_middleware));
        for &(path, health) in PATHS {
            let response = app.clone().oneshot(request(path)).await.unwrap();
            let expected = if health && !require_auth_for_health {
                StatusCode::NO_CONTENT
            } else {
                StatusCode::UNAUTHORIZED
            };
            assert_eq!(
                response.status(),
                expected,
                "{path}, require={require_auth_for_health}"
            );
        }
    }
}

#[tokio::test]
async fn auth_protected_request_requires_a_valid_key_and_stamps_identity() {
    let key = uuid::Uuid::new_v4().to_string();
    let config = MiddlewareAuthConfig {
        enabled: true,
        require_auth_for_health: true,
        api_keys: HashMap::from([(
            key.clone(),
            UserInfo {
                user_id: "contract-user".into(),
                tenant_id: Some("contract-tenant".into()),
                permissions: vec!["read".into()],
            },
        )]),
    };
    let app = Router::new()
        .fallback(
            |axum::Extension(user): axum::Extension<UserInfo>| async move {
                assert_eq!(user.user_id, "contract-user");
                assert_eq!(user.tenant_id.as_deref(), Some("contract-tenant"));
                assert_eq!(user.permissions, ["read"]);
                StatusCode::NO_CONTENT
            },
        )
        .layer(from_fn_with_state(config, auth_middleware));

    for path in ["/health", "/collections"] {
        let mut valid = request(path);
        valid.headers_mut().insert(
            axum::http::header::AUTHORIZATION,
            format!("Bearer {key}").parse().unwrap(),
        );
        assert_eq!(
            app.clone().oneshot(valid).await.unwrap().status(),
            StatusCode::NO_CONTENT
        );

        let mut invalid = request(path);
        invalid.headers_mut().insert(
            axum::http::header::AUTHORIZATION,
            uuid::Uuid::new_v4().to_string().parse().unwrap(),
        );
        assert_eq!(
            app.clone().oneshot(invalid).await.unwrap().status(),
            StatusCode::UNAUTHORIZED
        );
    }
}

#[tokio::test]
async fn disabled_auth_passes_protected_paths_without_identity() {
    let config = MiddlewareAuthConfig {
        enabled: false,
        require_auth_for_health: true,
        api_keys: HashMap::new(),
    };
    let app = Router::new()
        .fallback(|| async { StatusCode::NO_CONTENT })
        .layer(from_fn_with_state(config, auth_middleware));
    for &(path, _) in PATHS {
        assert_eq!(
            app.clone().oneshot(request(path)).await.unwrap().status(),
            StatusCode::NO_CONTENT
        );
    }
}

#[tokio::test]
async fn required_tls_certificate_is_exempted_only_on_health_paths() {
    let app = Router::new()
        .fallback(|| async { StatusCode::NO_CONTENT })
        .layer(from_fn_with_state(
            TlsClientCertState::new(TlsClientCertConfig::required()),
            tls_client_cert_middleware,
        ));
    for &(path, health) in PATHS {
        let expected = if health {
            StatusCode::NO_CONTENT
        } else {
            StatusCode::UNAUTHORIZED
        };
        assert_eq!(
            app.clone().oneshot(request(path)).await.unwrap().status(),
            expected,
            "{path}"
        );
    }
}

#[tokio::test]
async fn exhausted_rate_limits_exempt_health_only_when_configured() {
    for limit_health_endpoints in [false, true] {
        let state = Arc::new(RateLimitState::new(MiddlewareRateLimitConfig {
            enabled: true,
            max_requests: 0,
            global_max_requests: Some(0),
            window_duration: Duration::from_secs(60),
            limit_health_endpoints,
        }));
        let app = Router::new()
            .fallback(|| async { StatusCode::NO_CONTENT })
            .layer(from_fn_with_state(state, rate_limit_middleware));
        for &(path, health) in PATHS {
            let mut req = request(path);
            req.extensions_mut().insert(ConnectInfo(
                "198.51.100.7:1234".parse::<std::net::SocketAddr>().unwrap(),
            ));
            let expected = if health && !limit_health_endpoints {
                StatusCode::NO_CONTENT
            } else {
                StatusCode::TOO_MANY_REQUESTS
            };
            assert_eq!(
                app.clone().oneshot(req).await.unwrap().status(),
                expected,
                "{path}, limit={limit_health_endpoints}"
            );
        }
    }
}
