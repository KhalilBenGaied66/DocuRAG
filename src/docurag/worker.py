"""Ingestion worker: claims jobs from Postgres with SKIP LOCKED, so several can run side by side."""

from __future__ import annotations

import asyncio
import logging
from time import monotonic

from docurag.container import Container
from docurag.models import DocumentSpec

log = logging.getLogger(__name__)

STALE_AFTER_S = 1800  # a job 'running' this long is presumed lost with its worker
REQUEUE_EVERY_S = 60


async def run_worker(
    container: Container, *, once: bool = False, stop: asyncio.Event | None = None
) -> int:
    """Process jobs until stopped (or until the queue is empty when once=True)."""
    repo, pipeline, settings = container.repo, container.pipeline, container.settings
    processed = 0
    last_requeue = 0.0
    log.info("worker started")
    while not (stop and stop.is_set()):
        if monotonic() - last_requeue > REQUEUE_EVERY_S:
            requeued = await repo.requeue_stale_jobs(STALE_AFTER_S, settings.job_max_attempts)
            if requeued:
                log.warning("requeued %s stale job(s)", requeued)
            last_requeue = monotonic()

        job = await repo.claim_job()
        if job is None:
            if once:
                break
            await asyncio.sleep(settings.worker_poll_interval_s)
            continue

        job_id, attempts = job["id"], job["attempts"]
        spec = DocumentSpec.model_validate(job["spec"])
        try:
            result = await pipeline.ingest(spec)
        except Exception as exc:  # transient: embedding API, database
            retry = attempts < settings.job_max_attempts
            log.exception("job %s attempt %s failed (retry=%s)", job_id, attempts, retry)
            await repo.finish_job(job_id, error=str(exc), retry=retry)
        else:
            if result.status == "failed":  # permanent: retrying won't help
                await repo.finish_job(job_id, result=result.model_dump(mode="json"), error=result.error)
            else:
                await repo.finish_job(job_id, result=result.model_dump(mode="json"))
            log.info("job %s: %s %s", job_id, spec.source_uri, result.status)
        processed += 1
    return processed
