"""
Tests for the API layer.

Notes
-----
Covers: W3C TraceContextMiddleware (header generation, propagation, validation),
RFC 7807 error response format, and SSE headers middleware.

Uses Starlette's TestClient for ASGI-level testing — no network required.
"""
from __future__ import annotations

import json
import os
import re
import sys

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response
from starlette.routing import Route
from starlette.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))

from main import SseHeadersMiddleware, TraceContextMiddleware  # noqa: E402

_TRACEPARENT_RE = re.compile(
    r"^00-[0-9a-f]{32}-[0-9a-f]{16}-01$"
)


def _make_app_with_trace() -> tuple[Starlette, TestClient]:
    """
    Build a minimal ASGI app wrapped with TraceContextMiddleware.

    Returns
    -------
    tuple of Starlette and TestClient
        The configured application and a test client bound to it.

    Notes
    -----
    Exposes two routes: ``/echo`` returns the injected ``trace_id`` and
    ``span_id`` from the request scope, and ``/error`` raises a
    ``ValueError`` to test error propagation paths.
    """
    async def echo(request: Request) -> JSONResponse:
        return JSONResponse({
            "trace_id": request.scope.get("trace_id", ""),
            "span_id":  request.scope.get("span_id", ""),
        })

    async def error(request: Request) -> Response:
        raise ValueError("deliberate error")

    app = Starlette(routes=[
        Route("/echo", echo),
        Route("/error", error),
    ])
    app.add_middleware(TraceContextMiddleware)
    return app, TestClient(app, raise_server_exceptions=False)


class TestTraceContextMiddleware:
    """
    Tests for W3C Trace Context header generation, propagation, and validation.

    Notes
    -----
    Each test method is isolated via ``setup_method`` which creates a fresh
    client instance. The W3C traceparent format is ``00-<32hex>-<16hex>-01``.
    """

    def setup_method(self):
        """
        Initialise a fresh test client before each test method.

        Returns
        -------
        None

        Notes
        -----
        Creates a new ASGI app and TestClient to avoid state sharing between
        test methods in the class.
        """
        _, self.client = _make_app_with_trace()

    def test_generates_traceparent_header_when_absent(self):
        resp = self.client.get("/echo")
        assert "traceparent" in resp.headers
        assert _TRACEPARENT_RE.match(resp.headers["traceparent"])

    def test_traceparent_format_is_w3c_compliant(self):
        """
        Verify traceparent format: ``00-<32hex>-<16hex>-01``.

        Notes
        -----
        Checks all four components of the W3C traceparent header independently:
        version (``00``), trace-id (32 hex chars), parent-id (16 hex chars),
        and flags (``01``).
        """
        resp = self.client.get("/echo")
        parts = resp.headers["traceparent"].split("-")
        assert parts[0] == "00"
        assert len(parts[1]) == 32
        assert len(parts[2]) == 16
        assert parts[3] == "01"

    def test_propagates_valid_incoming_traceparent(self):
        """
        Verify that a valid incoming traceparent propagates the trace_id.

        Notes
        -----
        The trace_id from the incoming header should be preserved in the
        request scope and returned by the echo endpoint.
        """
        incoming = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
        resp = self.client.get("/echo", headers={"traceparent": incoming})
        body = resp.json()
        assert body["trace_id"] == "4bf92f3577b34da6a3ce929d0e0e4736"

    def test_generates_new_span_id_even_when_trace_propagated(self):
        """
        Verify that span-id is always fresh, not copied from the incoming header.

        Notes
        -----
        The span-id in the outgoing traceparent represents this service's span,
        not the parent's. It must differ from the parent span in the incoming
        header.
        """
        incoming = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
        resp = self.client.get("/echo", headers={"traceparent": incoming})
        parts = resp.headers["traceparent"].split("-")
        assert parts[2] != "00f067aa0ba902b7"
        assert len(parts[2]) == 16

    def test_ignores_invalid_traceparent_header(self):
        """
        Verify that a malformed traceparent triggers fresh trace generation.

        Notes
        -----
        Invalid traceparent headers must be silently ignored and a new valid
        traceparent generated, rather than raising an exception.
        """
        bad = "not-a-valid-traceparent"
        resp = self.client.get("/echo", headers={"traceparent": bad})
        assert _TRACEPARENT_RE.match(resp.headers["traceparent"])

    def test_ignores_all_zeros_trace_id(self):
        """
        Verify that an all-zeros trace-id (reserved per W3C spec) is rejected.

        Notes
        -----
        The W3C spec prohibits trace-ids composed entirely of zeros. The
        middleware must generate a fresh trace-id when this is detected.
        """
        bad = "00-00000000000000000000000000000000-00f067aa0ba902b7-01"
        resp = self.client.get("/echo", headers={"traceparent": bad})
        parts = resp.headers["traceparent"].split("-")
        assert parts[1] != "0" * 32

    def test_ignores_all_zeros_span_id(self):
        bad = "00-4bf92f3577b34da6a3ce929d0e0e4736-0000000000000000-01"
        resp = self.client.get("/echo", headers={"traceparent": bad})
        assert _TRACEPARENT_RE.match(resp.headers["traceparent"])

    def test_trace_id_injected_into_scope(self):
        resp = self.client.get("/echo")
        body = resp.json()
        assert len(body["trace_id"]) == 32
        assert all(c in "0123456789abcdef" for c in body["trace_id"])

    def test_span_id_injected_into_scope(self):
        resp = self.client.get("/echo")
        body = resp.json()
        assert len(body["span_id"]) == 16
        assert all(c in "0123456789abcdef" for c in body["span_id"])

    def test_each_request_gets_unique_trace(self):
        r1 = self.client.get("/echo")
        r2 = self.client.get("/echo")
        assert r1.headers["traceparent"] != r2.headers["traceparent"]


class TestSseHeadersMiddleware:
    """
    Tests for SSE-specific response headers injected by SseHeadersMiddleware.

    Notes
    -----
    The middleware should only inject ``cache-control: no-cache`` and
    ``x-accel-buffering: no`` headers on routes whose path ends with
    ``/run_sse`` (i.e. routes serving Server-Sent Events). Other routes
    must not be affected.
    """

    def setup_method(self):
        """
        Initialise a fresh test app with SseHeadersMiddleware before each test.

        Returns
        -------
        None

        Notes
        -----
        Exposes ``/run_sse`` (SSE content type) and ``/regular`` (JSON)
        to distinguish middleware behaviour between route types.
        """
        async def sse_stream(request: Request) -> PlainTextResponse:
            return PlainTextResponse("data: hello\n\n",
                                     media_type="text/event-stream")

        async def regular(request: Request) -> JSONResponse:
            return JSONResponse({"ok": True})

        app = Starlette(routes=[
            Route("/run_sse", sse_stream),
            Route("/regular", regular),
        ])
        app.add_middleware(SseHeadersMiddleware)
        self.client = TestClient(app)

    def test_sse_route_gets_cache_control_header(self):
        resp = self.client.get("/run_sse")
        assert resp.headers.get("cache-control") == "no-cache"

    def test_sse_route_gets_x_accel_buffering_header(self):
        resp = self.client.get("/run_sse")
        assert resp.headers.get("x-accel-buffering") == "no"

    def test_non_sse_route_not_affected(self):
        resp = self.client.get("/regular")
        assert resp.headers.get("cache-control") != "no-cache"


class TestRfc7807ErrorFormat:
    """
    Tests for RFC 7807 Problem Details format in error responses.

    Notes
    -----
    Tests the exception handlers directly by importing them and constructing
    minimal mock Request objects, rather than booting the full ADK app. This
    avoids the heavyweight ADK startup while still verifying the error payload
    schema.
    """

    def _make_mock_request(self, trace_id: str = "abc123") -> Request:
        """
        Create a minimal mock ASGI Request with a trace_id in scope.

        Parameters
        ----------
        trace_id : str, optional
            Trace ID to inject into the request scope. Defaults to ``"abc123"``.

        Returns
        -------
        Request
            Starlette Request object with the given trace_id in its ASGI scope.

        Notes
        -----
        The scope includes only the fields required by Starlette's Request
        constructor plus the custom ``trace_id`` field used by the error handler.
        """
        scope = {
            "type":    "http",
            "method":  "GET",
            "path":    "/api/v1/test",
            "query_string": b"",
            "headers": [],
            "trace_id": trace_id,
        }
        return Request(scope)

    def test_global_error_handler_returns_500(self):
        import asyncio

        from main import _global_error_handler
        req  = self._make_mock_request(trace_id="deadbeef" * 4)
        resp = asyncio.run(
            _global_error_handler(req, RuntimeError("boom"))
        )
        assert resp.status_code == 500

    def test_global_error_handler_content_type_is_problem_json(self):
        import asyncio

        from main import _global_error_handler
        req  = self._make_mock_request()
        resp = asyncio.run(
            _global_error_handler(req, RuntimeError("boom"))
        )
        assert "application/problem+json" in resp.media_type

    def test_global_error_handler_body_has_required_fields(self):
        import asyncio

        from main import _global_error_handler
        req  = self._make_mock_request(trace_id="a" * 32)
        resp = asyncio.run(
            _global_error_handler(req, RuntimeError("test error"))
        )
        body = json.loads(resp.body)
        for field in ("type", "title", "status", "detail", "trace_id"):
            assert field in body, f"RFC 7807 body missing field: {field}"

    def test_global_error_handler_uses_trace_id_from_scope(self):
        import asyncio

        from main import _global_error_handler
        trace = "a" * 32
        req   = self._make_mock_request(trace_id=trace)
        resp  = asyncio.run(
            _global_error_handler(req, RuntimeError("boom"))
        )
        body = json.loads(resp.body)
        assert body["trace_id"] == trace

    def test_global_error_handler_no_correlation_id_field(self):
        """
        Verify that the deprecated ``correlation_id`` field is absent.

        Notes
        -----
        The ``correlation_id`` field was replaced by ``trace_id`` to align
        with W3C Trace Context. Its presence would indicate a regression.
        """
        import asyncio

        from main import _global_error_handler
        req  = self._make_mock_request()
        resp = asyncio.run(
            _global_error_handler(req, RuntimeError("x"))
        )
        body = json.loads(resp.body)
        assert "correlation_id" not in body
