"""RQ worker entrypoint. Per the earlier job-queue decision (Python/RQ,
living alongside the GWR/XGBoost code) -- this process handles async work
enqueued elsewhere, distinct from the scheduled full-grid batch run in
run_pipeline.py.

src/workers/scoring.py's score_new_point(latitude, longitude) is the first
real job: on-demand scoring for a newly-registered UMKM point against the
grid the last batch run already scored (no GWR/XGBoost recomputation at
request time -- see that module's docstring). RQ jobs need no
worker-side registration; the backend enqueues by importing the function:

    from redis import Redis
    from rq import Queue
    from src.workers.scoring import score_new_point
    Queue("default", connection=Redis.from_url(REDIS_URL)).enqueue(
        score_new_point, latitude, longitude
    )

This worker process just needs `src` importable, which it already is.

Run with: python worker.py
"""

from redis import Redis
from rq import Worker
from src.config import settings

if __name__ == "__main__":
    conn = Redis.from_url(settings.redis_url)
    worker = Worker(["default"], connection=conn)
    worker.work()
