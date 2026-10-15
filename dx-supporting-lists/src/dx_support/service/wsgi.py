"""gunicorn entry point: `gunicorn dx_support.service.wsgi:app`.

Initialisation is synchronous and loud. A bad configuration, missing reference data, an unreachable result store or an unknown model name raises
here, the worker fails to boot, and the container exits with the reason in the log -- SageMaker then reports the deployment as failed within
seconds instead of waiting out the health-check timeout.
"""
from __future__ import annotations

import logging

from dx_support.logging_setup import configure_logging
from dx_support.service.app import create_app
from dx_support.service.runtime import ServiceRuntime
from dx_support.settings import Settings, environment_warnings

settings = Settings.from_env()
configure_logging(settings.log_level, settings.log_json)
logging.getLogger("dx_support.calls").setLevel(logging.INFO if settings.log_calls else logging.WARNING)
for _w in environment_warnings():
    logging.getLogger(__name__).warning(_w)
runtime = ServiceRuntime(settings).start()
app = create_app(runtime)
