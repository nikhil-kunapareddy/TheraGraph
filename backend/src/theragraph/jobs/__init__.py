"""Background job processing.

Expensive workflows -- ingestion, model training, full agent research runs -- must not
run inside an HTTP request. They are enqueued as durable jobs and executed by a separate
worker process.

See ``jobs/queue.py`` for the delivery guarantees and ``docs/design_decisions.md`` for
why this is a purpose-built queue rather than Celery.
"""

from theragraph.jobs.queue import InlineQueue, JobQueue, RedisQueue, build_queue
from theragraph.jobs.registry import TASKS, JobContext, task

__all__ = [
    "TASKS",
    "InlineQueue",
    "JobContext",
    "JobQueue",
    "RedisQueue",
    "build_queue",
    "task",
]
