"""WSGI entry point used by gunicorn inside the SageMaker container."""
from __future__ import annotations

import os

from ..logging_setup import configure_logging
from ..settings import Settings
from .app import create_app
from .container import build_container
from .health import check_health, run_dir


def _create():
    settings = Settings.load()
    configure_logging(os.environ.get("RULE_EXTRACTOR_LOG_LEVEL", "INFO"))
    directory = run_dir(settings)
    return create_app(build_container(settings, health=lambda: check_health(directory)))


app = _create()
