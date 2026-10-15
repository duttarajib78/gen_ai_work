"""Executable-rule engine: a small, fixed JSON grammar interpreted by hand-written, unit-tested code.

Nothing here uses ``eval``/``exec``. A malformed or adversarial rule can fail validation or evaluate to
the wrong boolean; it cannot read files, open sockets, or run arbitrary code.
"""
