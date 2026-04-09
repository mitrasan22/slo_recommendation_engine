"""
Pytest configuration and shared fixtures for the SLO recommendation engine test suite.

Notes
-----
Primary concern: the dependency agent tools use a module-level mutable dict
(``_graph_cache``) as an in-memory graph store. Without explicit reset between
tests, one test's ingest call leaves state that affects the next test, making
test ordering significant and hiding failures. The ``reset_graph_cache`` fixture
is ``autouse=True`` to prevent this.
"""
from __future__ import annotations

import collections
import importlib.metadata as _im

import pytest


def pytest_configure(config):  # noqa: ARG001
    """
    Patch ``importlib.metadata.packages_distributions`` for Python 3.14 compatibility.

    Parameters
    ----------
    config : pytest.Config
        Pytest configuration object (unused but required by the hook signature).

    Returns
    -------
    None

    Notes
    -----
    Python 3.14 + importlib_metadata 8.x: some distributions in the venv have
    ``METADATA`` files without a ``Name`` field. Python 3.14's stdlib
    ``packages_distributions()`` calls ``dist.metadata['Name']`` for every
    distribution, raising ``KeyError`` on malformed entries instead of skipping
    them. The ``transformers`` package imports ``packages_distributions()`` at
    module level, which crashes pytest collection before any test file is loaded.

    This patch reimplements ``packages_distributions()`` to silently skip any
    distribution whose metadata lacks a ``Name`` key.
    """
    def _safe_packages_distributions():
        pkg_to_dist: dict = collections.defaultdict(list)
        for dist in _im.distributions():
            try:
                name = dist.metadata["Name"]
            except (KeyError, Exception):
                continue
            for pkg in (dist.read_text("top_level.txt") or "").splitlines():
                pkg = pkg.strip()
                if pkg:
                    pkg_to_dist[pkg].append(name)
        return dict(pkg_to_dist)

    _im.packages_distributions = _safe_packages_distributions


@pytest.fixture(autouse=True)
def reset_graph_cache():
    """
    Reset the dependency agent's module-level graph cache before each test.

    Returns
    -------
    None

    Notes
    -----
    Marked ``autouse=True`` so every test gets a clean slate — callers do not
    need to request this fixture explicitly. The cache is cleared both before
    the test (setup) and after the test (teardown via the ``yield`` boundary)
    to ensure no state leaks in either direction.
    """
    from slo_engine.agents.dependency_agent.tools import tools as dep_tools
    dep_tools._graph_cache.clear()
    yield
    dep_tools._graph_cache.clear()
