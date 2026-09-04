"""Database connection lifecycle and error containment.

One engine per process, owned by the application lifespan, with a pooled
connection reused across requests. Nothing in the codebase opens its own
connection.

This module is also the boundary where SQLAlchemy exceptions stop. A raw
``SQLAlchemyError`` carries the failing SQL statement and, depending on the
driver, its bound parameters and the connection string — none of which may
reach an API consumer. Everything below translates driver failures into
:class:`DatabaseUnavailableError`, logging the real cause server-side under the
request's correlation id.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config.logging import get_logger
from app.core.exceptions import DatabaseUnavailableError, WeatherGPTError

logger = get_logger(__name__)


class Database:
    """Owns the async engine and hands out transactional sessions."""

    def __init__(
        self,
        url: str,
        *,
        echo: bool = False,
        pool_size: int = 5,
        max_overflow: int = 10,
        pool_timeout: float = 10.0,
        pool_recycle: int = 1800,
    ) -> None:
        self._engine: AsyncEngine = create_async_engine(
            url,
            echo=echo,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_timeout=pool_timeout,
            # Recycle before typical proxy/firewall idle timeouts so a pooled
            # connection is never handed out already dead.
            pool_recycle=pool_recycle,
            pool_pre_ping=True,
        )
        self._sessionmaker = async_sessionmaker(
            bind=self._engine, expire_on_commit=False, autoflush=False
        )

    @property
    def engine(self) -> AsyncEngine:
        return self._engine

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """A session wrapped in one transaction.

        Commits when the block completes, rolls back if it raises. Driver
        failures surface as :class:`DatabaseUnavailableError`; application
        errors raised inside the block propagate untouched, after the rollback.
        """
        session = self._sessionmaker()
        try:
            yield session
            await session.commit()
        except SQLAlchemyError as exc:
            await session.rollback()
            raise _translate(exc, operation="transaction") from exc
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    async def ping(self) -> bool:
        """Cheap liveness check for the readiness probe. Never raises.

        Catches broadly on purpose: connect-time failures can arrive as raw
        driver exceptions rather than SQLAlchemy ones (asyncpg raises
        ``InvalidCatalogNameError`` directly for a missing database), and a
        readiness probe that raises is worse than one that reports unhealthy.
        """
        try:
            async with self._engine.connect() as connection:
                await connection.execute(text("SELECT 1"))
        except Exception as exc:  # noqa: BLE001 - a probe reports, it never raises
            logger.warning(
                "database.ping_failed", extra={"error_type": type(exc).__name__}
            )
            return False
        return True

    async def dispose(self) -> None:
        """Close every pooled connection. Called once, at shutdown."""
        await self._engine.dispose()


def _translate(exc: Exception, *, operation: str) -> DatabaseUnavailableError:
    """Log the real failure and return a safe, generic application error.

    The exception message is deliberately *not* propagated into the API
    response: SQLAlchemy embeds the statement and parameters in it, and driver
    errors can name the database or host.
    """
    logger.exception(
        "database.operation_failed",
        extra={"operation": operation, "error_type": type(exc).__name__},
    )
    return DatabaseUnavailableError(details={"operation": operation})


@asynccontextmanager
async def translate_database_errors(operation: str) -> AsyncIterator[None]:
    """Contain database failures raised inside a read path.

    Catches broadly rather than only :class:`SQLAlchemyError`, because not every
    failure arrives wrapped: asyncpg raises its own exceptions at connect time,
    and one of those reaching the API would leak a host or database name. Our
    own structured errors pass through untouched, and the block wraps only the
    session and query — mapping happens outside it, so a bug in application code
    is never disguised as a database outage.
    """
    try:
        yield
    except WeatherGPTError:
        raise
    except Exception as exc:
        raise _translate(exc, operation=operation) from exc
