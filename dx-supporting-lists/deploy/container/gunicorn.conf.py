"""gunicorn settings for the SageMaker container.

ONE worker process, many threads, on purpose. The process owns the asyncio event loop that runs every model-calling job, the shared concurrency
limiter and the per-hour spend budget; several worker processes would each have their own, multiplying the real limits. The work is I/O-bound (waiting
for Bedrock and S3), so threads are plenty. The thread count leaves headroom beyond the request limit so /ping and over-limit rejections are always answered.
"""
import os
import sys

bind = "0.0.0.0:8080"
workers = 1
worker_class = "gthread"
threads = int(os.environ.get("DX_MAX_CONCURRENT_REQUESTS", "8")) + 4
timeout = 300                 # the worker heartbeat, not a request limit: long streaming requests are fine; start-up (loading the reference files) is not killed
graceful_timeout = 30         # SageMaker stops a container with SIGTERM and then kills it
keepalive = 75
worker_tmp_dir = "/dev/shm"
accesslog = None              # the application logs one structured line per request (ids and counts only; never request bodies)
errorlog = "-"
loglevel = "warning"
limit_request_line = 2048
limit_request_fields = 50
limit_request_field_size = 4096


def worker_exit(server, worker):
    """Runs in the worker as it exits: stop launching model calls and flush the run log."""
    wsgi = sys.modules.get("dx_support.service.wsgi")
    if wsgi is not None and hasattr(wsgi, "runtime"):
        wsgi.runtime.shutdown(timeout=20)
