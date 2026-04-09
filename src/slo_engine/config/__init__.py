"""
Configuration package for the SLO Recommendation Engine.

Notes
-----
Re-exports the singleton ``config`` object built from Dynaconf settings.
Import as ``from slo_engine.config import config``.
"""
from slo_engine.config.config import config

__all__ = ["config"]
