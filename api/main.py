"""
FastAPI application entry point for the SLO Recommendation Engine.

Notes
-----
Uses google.adk.cli.fast_api.get_fast_api_app to serve the ADK agent pipeline,
augmented with SLO-specific REST routes and middleware.

Middleware stack (outermost to innermost, i.e. last registered first to execute):
TraceContextMiddleware injects W3C Trace Context and structured request logging.
SseHeadersMiddleware prevents proxy buffering of SSE streams.
CORSMiddleware adds CORS headers.
RateLimitMiddleware enforces 60 req/min per IP using Redis or in-memory fallback.

Exception handlers conform to RFC 7807 Problem Details (application/problem+json).
The trace_id from the active W3C trace context is included in every error response.
"""
from __future__ import annotations

import os
import time
from argparse import ArgumentParser
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from google.adk.cli.fast_api import get_fast_api_app
from loguru import logger
from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from slo_engine.config import config
from slo_engine.utils.pii_scrubber import scrub as _scrub

load_dotenv()

logger = logger.bind(name="slo_engine.api.main")

BASE_DIR: Path = Path(os.path.abspath(os.path.dirname(__file__))).resolve().parent

_LOG_DIR = BASE_DIR / "logs"
_LOG_DIR.mkdir(exist_ok=True)
from loguru import logger as _root_logger


def _log_fmt(record: dict) -> str:
    """
    Format a Loguru log record into a structured string.

    Parameters
    ----------
    record : dict
        Loguru log record dictionary containing time, level, name, message,
        and extra fields.

    Returns
    -------
    str
        Formatted log line with timestamp, level, logger name, and message.

    Notes
    -----
    Curly braces and angle brackets in the message are escaped to prevent
    Loguru from interpreting them as markup tokens.
    """
    name = record["extra"].get("name") or record["name"]
    t    = record["time"].strftime("%H:%M:%S.%f")[:-3]
    lvl  = record["level"].name
    msg  = record["message"].replace("{", "{{").replace("}", "}}").replace("<", r"\<").replace(">", r"\>")
    return f"{t} | {lvl:<8} | {name:<50} | {msg}\n{{exception}}"


import datetime as _dt

_run_log = _LOG_DIR / f"run_{_dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

_root_logger.add(
    str(_run_log),
    level="DEBUG",
    format=_log_fmt,
    mode="w",
    enqueue=True,
    encoding="utf-8",
)
print(f"[SLO] Logging to: {_run_log}")
AGENT_DIR: Path = BASE_DIR / "src" / "slo_engine" / "agents"
SESSION_DB_URL: str = getattr(config, "session_db_uri", "sqlite+aiosqlite:///./slo_sessions.db")

app: FastAPI = get_fast_api_app(
    agents_dir=AGENT_DIR.as_posix(),
    session_service_uri=SESSION_DB_URL,
    web=True,
)

from api.middleware.rate_limit import RateLimitMiddleware

app.add_middleware(RateLimitMiddleware, limit=60)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SseHeadersMiddleware:
    """
    ASGI middleware that injects anti-buffering headers on SSE responses.

    Attributes
    ----------
    app : ASGIApp
        The wrapped ASGI application.

    Notes
    -----
    Adds ``Cache-Control: no-cache`` and ``X-Accel-Buffering: no`` to any
    response whose request path ends with ``/run_sse``, preventing Nginx and
    other reverse proxies from buffering the event stream.
    """

    def __init__(self, app: ASGIApp):
        """
        Initialise SseHeadersMiddleware.

        Parameters
        ----------
        app : ASGIApp
            The ASGI application to wrap.

        Notes
        -----
        Stores a reference to the inner application for delegation.
        """
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        """
        Process an ASGI request and inject SSE headers where applicable.

        Parameters
        ----------
        scope : Scope
            ASGI connection scope dictionary.
        receive : Receive
            ASGI receive callable.
        send : Send
            ASGI send callable.

        Returns
        -------
        None

        Notes
        -----
        Non-HTTP scopes are passed through without modification. For HTTP
        scopes with paths ending in ``/run_sse``, the ``http.response.start``
        message is intercepted and the anti-buffering headers are inserted.
        """
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")

        async def send_wrapper(message: Message):
            """
            Intercept response start messages to inject SSE headers.

            Parameters
            ----------
            message : Message
                ASGI message dictionary.

            Returns
            -------
            None

            Notes
            -----
            Only modifies ``http.response.start`` messages for SSE paths.
            """
            if message["type"] == "http.response.start" and path.endswith("/run_sse"):
                headers = MutableHeaders(raw=message["headers"])
                headers["Cache-Control"] = "no-cache"
                headers["X-Accel-Buffering"] = "no"
            await send(message)

        await self.app(scope, receive, send_wrapper)


app.add_middleware(SseHeadersMiddleware)


class TraceContextMiddleware:
    """
    W3C Trace Context propagation middleware.

    Reads the incoming ``traceparent`` header (if present and valid), extracts the
    trace-id, generates a new span-id for this hop, and propagates the context
    outward in the response ``traceparent`` header.

    The traceparent format is: ``{version}-{trace-id}-{parent-id}-{flags}`` where
    version is ``00``, trace-id is 32 hex chars, parent-id is 16 hex chars, and
    flags is ``01`` (sampled).

    Attributes
    ----------
    app : ASGIApp
        The wrapped ASGI application.

    Notes
    -----
    W3C Trace Context is used over X-Correlation-ID because it is an official
    W3C standard natively understood by Jaeger, Zipkin, Datadog, Honeycomb, and
    OpenTelemetry. Comet Opik traces tagged with the same trace-id share a single
    identifier across LLM calls and HTTP requests.

    Stores ``trace_id`` (32-char hex) and ``span_id`` (16-char hex) in the ASGI
    scope for downstream handlers and exception handlers to use.
    """

    _VERSION = "00"
    _FLAGS   = "01"

    def __init__(self, app: ASGIApp) -> None:
        """
        Initialise TraceContextMiddleware.

        Parameters
        ----------
        app : ASGIApp
            The ASGI application to wrap.

        Notes
        -----
        Stores a reference to the inner application for delegation.
        """
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """
        Process an ASGI request and inject W3C trace context.

        Parameters
        ----------
        scope : Scope
            ASGI connection scope dictionary.
        receive : Receive
            ASGI receive callable.
        send : Send
            ASGI send callable.

        Returns
        -------
        None

        Notes
        -----
        If a valid ``traceparent`` header is present, the existing trace-id is
        reused and a fresh span-id is generated for this hop. Otherwise, a new
        trace-id and span-id are generated. The outbound ``traceparent`` header
        is set on the response and the trace/span IDs are stored in the scope.
        """
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        raw_headers  = dict(scope.get("headers", []))
        incoming     = raw_headers.get(b"traceparent", b"").decode()

        if incoming and self._is_valid(incoming):
            trace_id = incoming.split("-")[1]
            span_id  = self._random_hex(8)
        else:
            trace_id = self._random_hex(16)
            span_id  = self._random_hex(8)

        traceparent = f"{self._VERSION}-{trace_id}-{span_id}-{self._FLAGS}"
        scope["trace_id"] = trace_id
        scope["span_id"]  = span_id
        start = time.perf_counter()

        async def _send(message: Message) -> None:
            """
            Intercept response start messages to attach the traceparent header.

            Parameters
            ----------
            message : Message
                ASGI message dictionary.

            Returns
            -------
            None

            Notes
            -----
            Also emits a structured log line with trace-id, span-id, HTTP method,
            path, status code, and elapsed milliseconds.
            """
            if message["type"] == "http.response.start":
                MutableHeaders(raw=message["headers"])["traceparent"] = traceparent
                logger.info(
                    "http | trace={} span={} method={} path={} status={} ms={:.1f}",
                    trace_id,
                    span_id,
                    scope.get("method", "-"),
                    _scrub(scope.get("path", "-")),
                    message.get("status", "-"),
                    (time.perf_counter() - start) * 1000,
                )
            await send(message)

        await self.app(scope, receive, _send)

    @staticmethod
    def _random_hex(n_bytes: int) -> str:
        """
        Generate a cryptographically random hex string.

        Parameters
        ----------
        n_bytes : int
            Number of random bytes to generate (output length is 2 * n_bytes).

        Returns
        -------
        str
            Lowercase hex string of length ``2 * n_bytes``.

        Notes
        -----
        Uses ``os.urandom`` which is suitable for cryptographic purposes.
        """
        return os.urandom(n_bytes).hex()

    @staticmethod
    def _is_valid(header: str) -> bool:
        """
        Validate a traceparent header value against the W3C spec.

        Parameters
        ----------
        header : str
            Raw traceparent header string.

        Returns
        -------
        bool
            ``True`` if the header conforms to W3C Trace Context format.

        Notes
        -----
        Rejects headers with all-zero trace-id or parent-id, which are
        reserved/invalid per the W3C specification.
        """
        parts = header.split("-")
        return (
            len(parts) == 4
            and parts[0] == "00"
            and len(parts[1]) == 32
            and len(parts[2]) == 16
            and parts[1] != "0" * 32
            and parts[2] != "0" * 16
        )


app.add_middleware(TraceContextMiddleware)


@app.exception_handler(RequestValidationError)
async def _validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """
    Handle Pydantic/FastAPI request validation errors with RFC 7807 format.

    Parameters
    ----------
    request : Request
        The FastAPI request object.
    exc : RequestValidationError
        The raised validation exception.

    Returns
    -------
    JSONResponse
        HTTP 422 response with ``application/problem+json`` content type.

    Notes
    -----
    The ``trace_id`` is extracted from the ASGI scope injected by
    ``TraceContextMiddleware``. Falls back to a freshly generated hex ID if
    not present.
    """
    trace_id = request.scope.get("trace_id", os.urandom(16).hex())
    logger.warning("validation error | trace={} path={}", trace_id, request.url.path)
    return JSONResponse(
        status_code=422,
        media_type="application/problem+json",
        content={
            "type":     "urn:slo-engine:errors:validation",
            "title":    "Request Validation Error",
            "status":   422,
            "detail":   exc.errors(),
            "instance": str(request.url),
            "trace_id": trace_id,
        },
    )


@app.exception_handler(Exception)
async def _global_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Handle all unhandled exceptions with RFC 7807 Problem Details format.

    Parameters
    ----------
    request : Request
        The FastAPI request object.
    exc : Exception
        The unhandled exception.

    Returns
    -------
    JSONResponse
        HTTP 500 response with ``application/problem+json`` content type.

    Notes
    -----
    PII scrubbing is applied to the exception message and request path before
    logging and before including in the response body. The ``trace_id`` is
    extracted from the ASGI scope injected by ``TraceContextMiddleware``.
    """
    trace_id = request.scope.get("trace_id", os.urandom(16).hex())
    safe_detail = _scrub(f"{type(exc).__name__}: {exc}")
    logger.error(
        "unhandled exception | trace={} path={} exc_type={} exc={}",
        trace_id,
        _scrub(request.url.path),
        type(exc).__name__,
        safe_detail,
    )
    return JSONResponse(
        status_code=500,
        media_type="application/problem+json",
        content={
            "type":     "urn:slo-engine:errors:internal",
            "title":    "Internal Server Error",
            "status":   500,
            "detail":   safe_detail,
            "instance": str(request.url),
            "trace_id": trace_id,
        },
    )


try:
    from api.routes import health, recommendations, reviews, services, slos

    app.include_router(health.router,           prefix="/api/v1", tags=["health"])
    app.include_router(services.router,         prefix="/api/v1", tags=["services"])
    app.include_router(slos.router,             prefix="/api/v1", tags=["slos"])
    app.include_router(recommendations.router,  prefix="/api/v1", tags=["recommendations"])
    app.include_router(reviews.router,          prefix="/api/v1", tags=["reviews"])
except ImportError as exc:
    logger.warning("Some API routes not available: {}", exc)

try:
    from api.routes.integrations import agent_card_router
    from api.routes.integrations import router as integrations_router

    app.include_router(integrations_router, prefix="/api/v1", tags=["integrations"])
    app.include_router(agent_card_router)
except ImportError as exc:
    logger.warning("Integration routes not available: {}", exc)


@app.get("/health", tags=["health"])
def health_check() -> dict:
    """
    Return a basic liveness check response.

    Returns
    -------
    dict
        Dictionary with keys ``status``, ``service``, and ``version``.

    Notes
    -----
    This endpoint is registered at ``/health`` (outside the ``/api/v1`` prefix)
    for use by container health probes and load balancer checks.
    """
    return {
        "status": "ok",
        "service": "SLO Recommendation Engine",
        "version": getattr(config, "app", {}).get("version", "1.0.0") if hasattr(config, "app") else "1.0.0",
    }


def main():
    """
    Parse CLI arguments and launch the Uvicorn ASGI server.

    Returns
    -------
    None

    Notes
    -----
    Accepts ``--debug`` / ``-d`` for auto-reload mode, ``--host``, ``--port`` / ``-p``,
    and ``--session-db`` to override the ADK session database URI. The debug flag
    can also be set via the ``config.app.debug`` Dynaconf key.
    """
    parser = ArgumentParser(description="Run the SLO Recommendation Engine API")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug mode with auto-reload")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", "-p", type=int, default=8000)
    parser.add_argument("--session-db", type=str, default=SESSION_DB_URL)
    args = parser.parse_args()

    debug = args.debug or getattr(getattr(config, "app", None), "debug", False)
    logger.info("Starting SLO Recommendation Engine API on {}:{}", args.host, args.port)
    logger.info("Agent directory: {}", AGENT_DIR.as_posix())
    logger.info("Session DB: {}", SESSION_DB_URL)

    uvicorn.run(
        "api.main:app",
        host=args.host,
        port=args.port,
        reload=debug,
        reload_dirs=[BASE_DIR.as_posix()],
        log_level="debug" if debug else "info",
    )


if __name__ == "__main__":
    main()
