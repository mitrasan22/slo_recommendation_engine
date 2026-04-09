"""
Async SQLAlchemy engine and session factory with Redis connection pool.

Notes
-----
Provides the PostgreSQL async engine, session factory, and context manager for
database access. Also provides a lazy-initialised Redis connection pool used
for caching and HITL pub/sub signalling.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import redis.asyncio as aioredis
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from slo_engine.config import config

_postgres_dsn    = str(config.postgres.dsn)
_postgres_pool   = int(getattr(config.postgres, "pool_size", 10))
_postgres_overflow = int(getattr(config.postgres, "max_overflow", 20))
_debug           = bool(getattr(getattr(config, "app", None), "debug", False))

engine: AsyncEngine = create_async_engine(
    _postgres_dsn,
    pool_size=_postgres_pool,
    max_overflow=_postgres_overflow,
    pool_pre_ping=True,
    echo=_debug,
)

AsyncSessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=engine,
    expire_on_commit=False,
    autoflush=False,
)


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Async context manager providing a managed database session.

    Returns
    -------
    AsyncGenerator of AsyncSession
        Yields an ``AsyncSession`` that commits on clean exit and rolls back
        on exception.

    Notes
    -----
    The session is automatically committed at the end of a clean ``async with``
    block and rolled back if any exception is raised. The session is always
    closed in the ``finally`` block to return it to the connection pool.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency providing an async database session per request.

    Returns
    -------
    AsyncGenerator of AsyncSession
        Yields a session from ``get_db_session`` for use as a FastAPI
        ``Depends`` injection target.

    Notes
    -----
    Delegates to ``get_db_session`` and is intended for use with
    ``fastapi.Depends(get_db)`` in route handler signatures.
    """
    async with get_db_session() as session:
        yield session


_redis_pool: aioredis.Redis | None = None


async def get_redis() -> aioredis.Redis:
    """
    Return the shared async Redis connection pool, initialising it if needed.

    Returns
    -------
    aioredis.Redis
        A connection pool configured with UTF-8 encoding and response decoding.

    Notes
    -----
    The pool is lazily initialised on first call and reused on subsequent calls.
    Connection URL is read from ``config.redis.url``. Pool capacity is capped
    at 50 connections. Use ``close_redis`` at application shutdown to cleanly
    drain the pool.
    """
    global _redis_pool
    if _redis_pool is None:
        redis_url = str(config.redis.url)
        _redis_pool = aioredis.from_url(
            redis_url,
            encoding="utf-8",
            decode_responses=True,
            max_connections=50,
        )
    return _redis_pool


async def close_redis() -> None:
    """
    Close the Redis connection pool and release all connections.

    Returns
    -------
    None

    Notes
    -----
    Sets the module-level ``_redis_pool`` to None after closing so that
    the next call to ``get_redis`` reinitialises a fresh pool. Should be
    called during FastAPI lifespan shutdown.
    """
    global _redis_pool
    if _redis_pool:
        await _redis_pool.aclose()
        _redis_pool = None
