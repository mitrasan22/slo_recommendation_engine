"""
Rate limiting middleware for the SLO Recommendation Engine API.

Notes
-----
Implements a fixed-window counter strategy keyed on (IP, minute-bucket).
A Redis backend is used when available for distributed correctness across
multiple workers or pods. An in-memory dictionary serves as a local dev
and single-process fallback when Redis is unavailable.

The middleware wraps the ASGI app at the outermost layer and applies to every
route automatically, including ADK's /run and /run_sse endpoints.

Requests that exceed the limit receive HTTP 429 with Retry-After,
X-RateLimit-Limit, X-RateLimit-Remaining, and X-RateLimit-Reset headers.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from typing import Any

from loguru import logger
from starlette.types import ASGIApp, Receive, Scope, Send

_EXEMPT_PATHS = {"/health", "/.well-known/agent.json"}

_DEFAULT_LIMIT = 60
_WINDOW_SECONDS = 60


class _InMemoryStore:
    """
    In-process fixed-window request counter.

    Attributes
    ----------
    _counts : dict
        Nested mapping of IP address to minute-bucket to request count.

    Notes
    -----
    Not thread-safe. Intended for local development and single-worker
    deployments only. Old buckets are evicted on each increment call to
    prevent unbounded memory growth.
    """

    def __init__(self) -> None:
        """
        Initialise the in-memory store with an empty counts dictionary.

        Notes
        -----
        Uses ``collections.defaultdict`` so missing IP keys are created
        automatically on first access.
        """
        self._counts: dict[str, dict[int, int]] = defaultdict(dict)

    def increment(self, ip: str, bucket: int) -> int:
        """
        Increment the request count for an IP in the current time bucket.

        Parameters
        ----------
        ip : str
            Client IP address string.
        bucket : int
            Integer minute-bucket (current Unix time divided by window size).

        Returns
        -------
        int
            Updated request count for this IP in this bucket.

        Notes
        -----
        Buckets older than the previous window are evicted to bound memory use.
        """
        counts = self._counts[ip]
        for old in [b for b in counts if b < bucket - 1]:
            del counts[old]
        counts[bucket] = counts.get(bucket, 0) + 1
        return counts[bucket]


class _RedisStore:
    """
    Redis-backed fixed-window request counter.

    Attributes
    ----------
    _r : Any
        Synchronous Redis client instance.

    Notes
    -----
    Uses a pipeline with INCR + EXPIRE to atomically increment the counter
    and set a TTL slightly longer than the window to handle clock skew.
    Keys follow the pattern ``rl:{ip}:{bucket}``.
    """

    def __init__(self, redis_client: Any) -> None:
        """
        Initialise the Redis store with a connected client.

        Parameters
        ----------
        redis_client : Any
            A synchronous Redis client (e.g. from ``redis.from_url``).

        Notes
        -----
        The client must already be connected and pingable before being
        passed to this constructor.
        """
        self._r = redis_client

    def increment(self, ip: str, bucket: int) -> int:
        """
        Atomically increment the request count for an IP in the current bucket.

        Parameters
        ----------
        ip : str
            Client IP address string.
        bucket : int
            Integer minute-bucket (current Unix time divided by window size).

        Returns
        -------
        int
            Updated request count for this IP in this bucket.

        Notes
        -----
        Uses a Redis pipeline to execute INCR and EXPIRE atomically. The TTL
        is set to the window length plus 5 seconds to tolerate minor clock drift.
        """
        key = f"rl:{ip}:{bucket}"
        pipe = self._r.pipeline()
        pipe.incr(key)
        pipe.expire(key, _WINDOW_SECONDS + 5)
        count, _ = pipe.execute()
        return count


class RateLimitMiddleware:
    """
    Fixed-window rate limiter as a pure Starlette ASGI middleware.

    Attributes
    ----------
    app : ASGIApp
        The wrapped ASGI application.
    limit : int
        Maximum number of requests allowed per IP per minute window.

    Notes
    -----
    Attempts to use a Redis backend on initialisation; falls back to an
    in-memory store if Redis is unreachable. Paths listed in ``_EXEMPT_PATHS``
    are passed through without rate limit checks. Allowed requests receive
    informational X-RateLimit-* headers on the response.
    """

    def __init__(self, app: ASGIApp, limit: int = _DEFAULT_LIMIT) -> None:
        """
        Initialise RateLimitMiddleware.

        Parameters
        ----------
        app : ASGIApp
            The ASGI application to wrap.
        limit : int, optional
            Maximum requests per minute per IP. Defaults to 60.

        Notes
        -----
        Calls ``_try_redis`` during initialisation to probe Redis availability
        and select the appropriate backend store.
        """
        self.app = app
        self.limit = limit
        self._store: _InMemoryStore | _RedisStore = _InMemoryStore()
        self._try_redis()

    def _try_redis(self) -> None:
        """
        Attempt to connect to Redis and switch the store to a Redis backend.

        Returns
        -------
        None

        Notes
        -----
        Reads the Redis URL from ``config.redis.url``, falling back to
        ``redis://localhost:6379/0``. If the connection attempt fails for any
        reason the in-memory store is retained without raising an exception.
        """
        try:
            import redis as _redis
            from slo_engine.config import config as _cfg
            redis_url = getattr(getattr(_cfg, "redis", None), "url", None) or "redis://localhost:6379/0"
            client = _redis.from_url(redis_url, socket_connect_timeout=1)
            client.ping()
            self._store = _RedisStore(client)
            logger.info("RateLimitMiddleware: using Redis backend ({})", redis_url)
        except Exception:
            logger.info("RateLimitMiddleware: Redis unavailable — using in-memory backend")

    @staticmethod
    def _get_ip(scope: Scope) -> str:
        """
        Extract the client IP address from the ASGI scope.

        Parameters
        ----------
        scope : Scope
            ASGI connection scope dictionary.

        Returns
        -------
        str
            Client IP address string, or ``"unknown"`` if not determinable.

        Notes
        -----
        Prefers the ``client`` tuple in the scope. Falls back to parsing the
        first address from the ``X-Forwarded-For`` header when behind a proxy.
        """
        client = scope.get("client")
        if client:
            return client[0]
        for name, value in scope.get("headers", []):
            if name == b"x-forwarded-for":
                return value.decode().split(",")[0].strip()
        return "unknown"

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """
        Process an ASGI request and enforce the rate limit.

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
        Non-HTTP scopes and exempt paths are passed through immediately.
        If the rate limit is exceeded, a 429 response is returned directly
        without delegating to the inner application. On store errors the
        request is allowed through and a warning is logged.
        """
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if path in _EXEMPT_PATHS:
            await self.app(scope, receive, send)
            return

        ip     = self._get_ip(scope)
        now    = int(time.time())
        bucket = now // _WINDOW_SECONDS
        reset  = (bucket + 1) * _WINDOW_SECONDS

        try:
            count = self._store.increment(ip, bucket)
        except Exception as exc:
            logger.warning("Rate limit store error: {} — allowing request", exc)
            await self.app(scope, receive, send)
            return

        remaining = max(0, self.limit - count)

        if count > self.limit:
            retry_after = reset - now
            body = json.dumps({
                "detail": "Rate limit exceeded. Max 60 requests/minute.",
                "retry_after_seconds": retry_after,
            }).encode()
            await send({
                "type": "http.response.start",
                "status": 429,
                "headers": [
                    (b"content-type",        b"application/json"),
                    (b"content-length",      str(len(body)).encode()),
                    (b"retry-after",         str(retry_after).encode()),
                    (b"x-ratelimit-limit",   str(self.limit).encode()),
                    (b"x-ratelimit-remaining", b"0"),
                    (b"x-ratelimit-reset",   str(reset).encode()),
                ],
            })
            await send({"type": "http.response.body", "body": body})
            return

        async def send_with_headers(message: dict) -> None:
            """
            Inject rate limit informational headers into allowed responses.

            Parameters
            ----------
            message : dict
                ASGI message dictionary.

            Returns
            -------
            None

            Notes
            -----
            Only modifies ``http.response.start`` messages. Appends
            X-RateLimit-Limit, X-RateLimit-Remaining, and X-RateLimit-Reset
            to the outgoing response headers.
            """
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers += [
                    (b"x-ratelimit-limit",     str(self.limit).encode()),
                    (b"x-ratelimit-remaining", str(remaining).encode()),
                    (b"x-ratelimit-reset",     str(reset).encode()),
                ]
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_headers)
