#!/bin/sh
# SageMaker starts a hosting container as `docker run <image> serve`. Anything else runs the CLI, which is how the same image does batch builds
# (for example a SageMaker Processing job with ContainerArguments ["build", "--for-service", ...]).
set -eu
case "${1:-serve}" in
  serve)
    exec gunicorn --config /opt/program/gunicorn.conf.py "dx_support.service.wsgi:app"
    ;;
  dx-support)
    shift
    exec dx-support "$@"
    ;;
  *)
    exec dx-support "$@"
    ;;
esac
