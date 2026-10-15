#!/bin/sh
# SageMaker runs `docker run <image> serve`. Other commands are for local debugging:
#   docker run ... <image> cli plan        -> python -m rule_extractor plan
set -eu
cmd="${1:-serve}"
case "$cmd" in
  serve)  shift || true; exec python -m rule_extractor serve "$@" ;;
  worker) shift; exec python -m rule_extractor worker "$@" ;;
  cli)    shift; exec python -m rule_extractor "$@" ;;
  *)      exec "$@" ;;
esac
