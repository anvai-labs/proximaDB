//! TD-SECSCAN-1 §4: AWS environment detection must use IMDSv2.
//!
//! The detector issued a bare token-less IMDSv1 GET for `instance-id` while the
//! spot-eviction watch (`sst::warming`) already used IMDSv2. IMDSv1 is the
//! SSRF-reachable variant AWS recommends disabling, and on an instance with
//! `HttpTokens=required` a v1 probe gets 401 — so the detector also
//! mis-reported a hardened EC2 host as "not AWS".
//!
//! The stand-in below behaves like an IMDSv2-required instance: a token-less
//! GET is refused with 401; a GET carrying the PUT-issued session token is
//! served.

use httpmock::Method::{GET, PUT};
use httpmock::MockServer;
use httpmock::prelude::HttpMockRequest;
use proximadb::deployment::discovery::environment_detector::aws_imds_reachable;

const TOKEN: &str = "imdsv2-session-token";
const INSTANCE_ID_PATH: &str = "/latest/meta-data/instance-id";

/// Matches a request that carries no IMDSv2 session-token header.
fn lacks_session_token(req: &HttpMockRequest) -> bool {
    !req.headers.as_ref().is_some_and(|headers| {
        headers
            .iter()
            .any(|(name, _)| name.eq_ignore_ascii_case("x-aws-ec2-metadata-token"))
    })
}

#[tokio::test]
async fn aws_detection_uses_imdsv2_session_token() {
    let imds = MockServer::start_async().await;
    let token_mock = imds
        .mock_async(|when, then| {
            when.method(PUT)
                .path("/latest/api/token")
                .header_exists("X-aws-ec2-metadata-token-ttl-seconds");
            then.status(200).body(TOKEN);
        })
        .await;
    let with_token = imds
        .mock_async(|when, then| {
            when.method(GET)
                .path(INSTANCE_ID_PATH)
                .header("X-aws-ec2-metadata-token", TOKEN);
            then.status(200).body("i-0123456789abcdef0");
        })
        .await;
    let tokenless = imds
        .mock_async(|when, then| {
            when.method(GET)
                .path(INSTANCE_ID_PATH)
                .matches(lacks_session_token);
            then.status(401);
        })
        .await;

    let client = reqwest::Client::builder()
        .no_proxy()
        .build()
        .expect("http client");
    let reachable = aws_imds_reachable(&client, &imds.base_url()).await;

    assert_eq!(
        tokenless.hits_async().await,
        0,
        "detector issued a token-less IMDSv1 GET"
    );
    assert_eq!(token_mock.hits_async().await, 1, "no IMDSv2 token PUT");
    assert_eq!(with_token.hits_async().await, 1, "no token-bearing GET");
    assert!(reachable, "an IMDSv2-required instance must be detected");
}

#[tokio::test]
async fn aws_detection_never_falls_back_to_imdsv1() {
    // Token endpoint refuses (e.g. hop limit exceeded): the probe must report
    // "not reachable" rather than retry token-less.
    let imds = MockServer::start_async().await;
    imds.mock_async(|when, then| {
        when.method(PUT).path("/latest/api/token");
        then.status(403);
    })
    .await;
    let any_get = imds
        .mock_async(|when, then| {
            when.method(GET).path(INSTANCE_ID_PATH);
            then.status(200).body("i-0123456789abcdef0");
        })
        .await;

    let client = reqwest::Client::builder()
        .no_proxy()
        .build()
        .expect("http client");
    let reachable = aws_imds_reachable(&client, &imds.base_url()).await;

    assert_eq!(
        any_get.hits_async().await,
        0,
        "fell back to a token-less GET"
    );
    assert!(!reachable);
}
