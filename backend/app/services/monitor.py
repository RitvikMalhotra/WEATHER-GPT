"""The thing that makes a watch automatic.

One asyncio task, owned by the application lifespan, that periodically asks the
subscription service to refresh the watches that have gone longest without a
look. That is the entire mechanism.

It is deliberately the smallest reliable thing that demonstrates automatic
evaluation. It introduces no broker, no worker process and no new dependency,
and it holds no alert logic of its own — it calls the same service the HTTP
refresh endpoint calls, which calls the same weather pipeline every other
request calls.

**Replacing it is meant to be easy.** The seam is
:meth:`SubscriptionService.evaluate_due`: a production deployment points a real
scheduler (a task queue, a cron worker, a Kubernetes CronJob) at that method
and switches this loop off with ``ALERT_MONITOR_ENABLED=false``. Nothing else
changes, because nothing else knows this loop exists.

Known limitation, stated plainly: an in-process loop runs once per process, so
a multi-worker deployment would evaluate each watch once per worker. That is
wasteful rather than wrong — evaluation is idempotent and the alert store
deduplicates — but it is the reason this is a prototype mechanism and not a
production one.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from app.config.logging import get_logger
from app.services.subscriptions import SubscriptionService

logger = get_logger(__name__)


class AlertMonitor:
    """Periodically re-evaluates watched locations."""

    def __init__(
        self,
        subscriptions: SubscriptionService,
        *,
        interval_seconds: float = 900.0,
        batch_size: int = 50,
        start_delay_seconds: float = 5.0,
    ) -> None:
        self._subscriptions = subscriptions
        self._interval = max(30.0, interval_seconds)
        self._batch_size = batch_size
        # A short delay so startup — and the readiness probe with it — is not
        # competing with a sweep for the connection pool.
        self._start_delay = max(0.0, start_delay_seconds)
        self._task: asyncio.Task[None] | None = None
        self._stopping = asyncio.Event()
        self.last_sweep_at: datetime | None = None
        self.last_sweep_count: int = 0

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self) -> None:
        """Begin sweeping, unless there is nothing to sweep into."""
        if not self._subscriptions.enabled:
            logger.info("monitor.disabled", extra={"reason": "persistence_disabled"})
            return
        if self.running:
            return
        self._stopping.clear()
        self._task = asyncio.create_task(self._run(), name="weathergpt-alert-monitor")
        logger.info("monitor.started", extra={"interval_seconds": self._interval})

    async def stop(self) -> None:
        """Stop sweeping and wait for the current pass to finish."""
        if self._task is None:
            return
        self._stopping.set()
        self._task.cancel()
        try:
            await self._task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001 - shutdown path
            pass
        self._task = None
        logger.info("monitor.stopped")

    async def sweep_once(self) -> int:
        """Evaluate one batch of due watches. Never raises.

        A failing sweep must not kill the loop: the next one may well succeed,
        and a monitor that dies silently is worse than one that logs and
        retries.
        """
        try:
            evaluated = await self._subscriptions.evaluate_due(limit=self._batch_size)
        except Exception:  # noqa: BLE001 - a bad sweep is not a fatal condition
            logger.warning("monitor.sweep_failed", exc_info=True)
            return 0
        self.last_sweep_at = datetime.now(timezone.utc)
        self.last_sweep_count = evaluated
        return evaluated

    async def _run(self) -> None:
        try:
            await asyncio.wait_for(self._stopping.wait(), timeout=self._start_delay)
            return  # asked to stop before the first sweep
        except asyncio.TimeoutError:
            pass

        while not self._stopping.is_set():
            await self.sweep_once()
            try:
                # Waiting on the stop event rather than sleeping means shutdown
                # is immediate instead of taking up to a full interval.
                await asyncio.wait_for(self._stopping.wait(), timeout=self._interval)
            except asyncio.TimeoutError:
                continue
