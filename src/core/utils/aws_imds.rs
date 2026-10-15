// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! The ONE AWS instance-metadata (IMDS) GET used by the server — always
//! IMDSv2 (session token), never the SSRF-reachable token-less IMDSv1
//! (TD-SECSCAN-1 §4).
//!
//! IMDS has no TLS variant on any provider, so the endpoint is plain HTTP by
//! design; the hardening lever is the PUT-issued session token, which a
//! simple SSRF GET cannot obtain. Callers: environment detection
//! (`deployment::discovery`) and the spot-eviction watch (`sst::warming`).

/// AWS IMDS link-local base URL.
pub const AWS_IMDS_BASE: &str = "http://169.254.169.254";

/// Session-token lifetime requested from IMDS. Each call fetches a fresh
/// token, so the shortest sensible TTL is fine.
const TOKEN_TTL_SECS: &str = "300";

/// GET `{base}{path}` with an IMDSv2 session token: `PUT /latest/api/token`,
/// then the GET carrying `X-aws-ec2-metadata-token`. A failed or non-success
/// token request is an error — this never falls back to a token-less GET.
pub async fn aws_imdsv2_get(
    client: &reqwest::Client,
    base: &str,
    path: &str,
) -> Result<reqwest::Response, String> {
    let token_resp = client
        .put(format!("{base}/latest/api/token"))
        .header("X-aws-ec2-metadata-token-ttl-seconds", TOKEN_TTL_SECS)
        .send()
        .await
        .map_err(|e| e.to_string())?;
    if !token_resp.status().is_success() {
        return Err(format!(
            "IMDSv2 token request failed: HTTP {}",
            token_resp.status()
        ));
    }
    let token = token_resp.text().await.map_err(|e| e.to_string())?;
    client
        .get(format!("{base}{path}"))
        .header("X-aws-ec2-metadata-token", token.trim())
        .send()
        .await
        .map_err(|e| e.to_string())
}
