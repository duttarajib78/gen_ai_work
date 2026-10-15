"""Gunicorn settings for the SageMaker container.

The work is I/O bound (waiting on Amazon Bedrock), so threads rather than many
processes. A request can legitimately hold a thread for minutes in streaming
mode, hence the long worker timeout (it only applies to the worker's heartbeat
with gthread, not to request threads) and graceful shutdown window.

Each worker process loads its own reference data and clients, so a start-up
problem (bad configuration, failed integrity check) makes the worker fail to
boot and the container exit, which surfaces clearly in the endpoint's logs.
"""
import os

bind = "0.0.0.0:8080"
worker_class = "gthread"
workers = int(os.environ.get("AUDIT_WORKERS", "2"))
threads = int(os.environ.get("AUDIT_THREADS", "8"))
timeout = 600
graceful_timeout = 120
keepalive = 75
max_requests = 5000          # recycle workers occasionally to bound any slow leak
max_requests_jitter = 500
worker_tmp_dir = "/dev/shm"
accesslog = None             # the application logs one structured line per request; URLs carry no data anyway
errorlog = "-"
loglevel = os.environ.get("AUDIT_LOG_LEVEL", "info").lower()
limit_request_line = 4094
limit_request_fields = 50
limit_request_field_size = 8190
