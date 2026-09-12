"""RQ worker entrypoint. Per the earlier job-queue decision (Python/RQ,
living alongside the GWR/XGBoost code) -- this process handles async work
enqueued elsewhere (e.g. a future live trigger for on-demand scoring of a
newly-registered UMKM point, distinct from the scheduled full-grid batch
run in run_pipeline.py).

NOT YET given any real jobs to process -- src/workers/scoring.py's
run_scoring_batch() is still an intentional NotImplementedError stub (see
TODO.md history). This entrypoint exists so the worker PROCESS is real and
startable; wiring real jobs into it is separate, future work.

Run with: python worker.py
"""

from redis import Redis
from rq import Worker
from src.config import settings

if __name__ == "__main__":
    conn = Redis.from_url(settings.redis_url)
    worker = Worker(["default"], connection=conn)
    worker.work()
