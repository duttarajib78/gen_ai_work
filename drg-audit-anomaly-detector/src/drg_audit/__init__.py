"""DRG audit: transformer-based AR-DRG billing anomaly detection.

The package is split so that the real-time endpoint only needs the lightweight modules
(``features``, ``serving``, ``inference``, ``models``, ``artifacts``). Training, data and
post-processing modules pull in pandas / scikit-learn and are never imported by the endpoint.
"""

__version__ = "1.0.0"
