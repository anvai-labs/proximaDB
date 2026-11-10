class TestSwallowDiseaseGuard:
    """Ranked cleanup item 2/11: HTTP 200 with {"success": false} must raise,
    not return the failure body as if it were data (the long-documented
    swallow disease — 97 REST protocol methods, one check before this)."""

    def test_success_false_raises(self):
        class _FakeResponse:
            status_code = 200
            headers = {"content-type": "application/json"}
            text = '{"success": false, "error_message": "boom"}'

            def json(self):
                return {"success": False, "error_message": "boom"}

        resp = _FakeResponse()
        # Directly test the parse-check logic via a live _make_request would
        # need a transport mock; instead pin the helper + the shape gate.
        from proximadb_sdk.protocols.rest_sync import _success_false_is_partial

        assert not _success_false_is_partial(
            {"success": False, "error_message": "boom"}
        )
        # Batch shapes bypass (the caller aggregates partial results)
        assert _success_false_is_partial(
            {"success": False, "results": [], "successful_count": 0}
        )
        assert _success_false_is_partial({"success": False, "records": [{}]})

    def test_non_json_body_passes_through(self):
        from proximadb_sdk.protocols.rest_sync import _success_false_is_partial

        # Non-dict / missing keys: no raise
        assert not _success_false_is_partial({})
        assert not _success_false_is_partial({"other": 1})


class TestCentralCheckViaMockTransport:
    """The central _make_request check, exercised through a REAL transport
    mock (review round-1: the helper-only tests couldn't see regressions in
    the check itself — content-type gate, is-False discrimination, raise)."""

    def _client_with_transport(
        self, body: dict, content_type: str = "application/json"
    ):
        import httpx

        from proximadb_sdk.config import ClientConfig
        from proximadb_sdk.protocols.rest_sync import ProximaDBClient

        client = ProximaDBClient.__new__(ProximaDBClient)
        client.config = ClientConfig(url="http://test")
        client._http_client = httpx.Client(
            transport=httpx.MockTransport(
                lambda req: httpx.Response(
                    200, headers={"content-type": content_type}, json=body
                )
            ),
            base_url="http://test",
        )
        return client

    def test_200_success_false_raises_through_make_request(self):
        from proximadb_sdk.protocols.rest_sync import ProximaDBError

        client = self._client_with_transport(
            {"success": False, "error_message": "write rejected"}
        )
        try:
            client._make_request("GET", "/v2/x")
            raise AssertionError("must raise")
        except ProximaDBError as e:
            assert "write rejected" in str(e)

    def test_200_success_true_passes(self):
        client = self._client_with_transport({"success": True, "data": []})
        resp = client._make_request("GET", "/v2/x")
        assert resp.status_code == 200

    def test_200_partial_ingest_bypasses(self):
        """The observability partial-ingest shape (ingested/failed) must NOT
        raise — the caller aggregates the partial counts."""
        client = self._client_with_transport(
            {"success": False, "ingested": 3, "failed": 1}
        )
        resp = client._make_request("GET", "/v2/x")
        assert resp.status_code == 200

    def test_nested_error_message_extracted(self):
        from proximadb_sdk.protocols.rest_sync import ProximaDBError

        client = self._client_with_transport(
            {"success": False, "error": {"code": "GRAPH_001", "message": "bad pattern"}}
        )
        try:
            client._make_request("GET", "/v2/x")
            raise AssertionError("must raise")
        except ProximaDBError as e:
            assert "bad pattern" in str(e)

    def test_charset_variant_checked(self):
        from proximadb_sdk.protocols.rest_sync import ProximaDBError

        client = self._client_with_transport(
            {"success": False, "error_message": "x"},
            content_type="application/json; charset=utf-8",
        )
        try:
            client._make_request("GET", "/v2/x")
            raise AssertionError("charset variant must still raise")
        except ProximaDBError:
            pass
