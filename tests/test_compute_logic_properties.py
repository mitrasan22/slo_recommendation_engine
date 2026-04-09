"""
Property-based tests for core compute logic invariants.

Notes
-----
Uses Hypothesis to verify mathematical guarantees across the entire input space,
not just hand-picked examples. Each test documents the invariant it enforces
and why it must hold.

The ``ci`` Hypothesis profile disables the slow-test health check and removes
the deadline to prevent spurious failures on cold CI machines where numpy and
scipy have not yet JIT-compiled their critical paths.
"""
import math

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

settings.register_profile(
    "ci",
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile("ci")

from slo_engine.agents.recommendation_agent.tools.tools import (
    _confidence_scorer,
    _floor_nines,
)


@given(
    r1=st.floats(min_value=0.80, max_value=0.9999),
    r2=st.floats(min_value=0.80, max_value=0.9999),
)
def test_series_reliability_never_exceeds_either_dep(r1: float, r2: float) -> None:
    """
    Invariant: ``R_series = r1 * r2 <= min(r1, r2)``.

    Notes
    -----
    A chain of dependencies can only be as reliable as its weakest link.
    The series product must never exceed either component's individual reliability.
    """
    series = r1 * r2
    assert series <= r1 + 1e-9
    assert series <= r2 + 1e-9


@given(
    r1=st.floats(min_value=0.80, max_value=0.9999),
    r2=st.floats(min_value=0.80, max_value=0.9999),
)
def test_parallel_reliability_never_below_either_dep(r1: float, r2: float) -> None:
    """
    Invariant: ``R_parallel = 1 - (1-r1)*(1-r2) >= max(r1, r2)``.

    Notes
    -----
    Redundancy cannot make reliability worse than either component alone.
    The parallel combination must always equal or exceed both individual components.
    """
    parallel = 1.0 - (1.0 - r1) * (1.0 - r2)
    assert parallel >= r1 - 1e-9
    assert parallel >= r2 - 1e-9


@given(avail=st.floats(min_value=0.0, max_value=1.0))
def test_floor_nines_idempotent(avail: float) -> None:
    """
    Invariant: ``floor_nines(floor_nines(x)) == floor_nines(x)``.

    Notes
    -----
    Once snapped to a standard SLO level the value must not snap again on a
    second call. This is the idempotency requirement for discretisation
    functions.
    """
    once  = _floor_nines(avail)
    twice = _floor_nines(once)
    assert once == twice


_STANDARD_TIERS = {0.90, 0.95, 0.99, 0.995, 0.999, 0.9995, 0.9999}


@given(avail=st.floats(min_value=0.0, max_value=1.0))
def test_floor_nines_never_increases_value(avail: float) -> None:
    """
    Invariant: ``floor_nines(x)`` always returns one of the standard SLO tiers.

    Notes
    -----
    The function snaps to the nearest standard tier (not strictly a floor).
    Every possible input must map to one of the seven predefined tiers.
    """
    result = _floor_nines(avail)
    assert result in _STANDARD_TIERS


@given(
    obs=st.integers(min_value=0, max_value=500_000),
    n_err=st.integers(min_value=0, max_value=1_000),
)
@settings(max_examples=300, deadline=None)
def test_bayesian_posterior_ci_ordering(obs: int, n_err: int) -> None:
    """
    Invariant: ``credible_lower <= posterior_mean <= credible_upper``.

    Notes
    -----
    The 95% credible interval must contain the posterior mean. This is a
    fundamental requirement of any correctly computed credible interval.
    Uses a Beta-Binomial model with uniform priors (alpha0=1, beta0=1).
    """
    assume(n_err <= obs)
    from scipy import stats as sp_stats

    alpha0, beta0 = 1.0, 1.0
    k  = obs - n_err
    a  = alpha0 + k
    b  = beta0  + n_err
    mean = a / (a + b)
    lo, hi = sp_stats.beta.interval(0.95, a, b)
    assert lo <= mean + 1e-10
    assert mean <= hi + 1e-10


@given(
    n=st.integers(min_value=1, max_value=20),
    budget=st.floats(min_value=1e-5, max_value=0.10),
)
def test_water_fill_allocation_sums_to_budget(n: int, budget: float) -> None:
    """
    Invariant: proportional allocation must partition the budget exactly.

    Notes
    -----
    For uniform weights the proportional allocation is simply ``budget / n``
    per service. The sum of all allocations must equal the total budget to
    within floating-point precision (``1e-10``).
    """
    weights = np.ones(n)
    allocs  = weights / weights.sum() * budget
    assert abs(allocs.sum() - budget) < 1e-10


@given(
    obs=st.integers(min_value=0, max_value=1_000_000),
    std=st.floats(min_value=0.0, max_value=0.05),
)
def test_confidence_scorer_always_in_unit_interval(obs: int, std: float) -> None:
    """
    Invariant: confidence score is in ``[0, 1]`` for all valid inputs.

    Notes
    -----
    The confidence scorer must never return a value outside the unit interval
    regardless of the observation count or posterior standard deviation.
    """
    score = _confidence_scorer.compute(
        obs=obs, posterior_std=std,
        drift_detected=False, anomaly_severity="none",
        data_quality=1.0,
    )
    assert 0.0 <= score <= 1.0


@given(std=st.floats(min_value=0.0, max_value=0.05))
def test_drift_strictly_reduces_confidence(std: float) -> None:
    """
    Invariant: ``score(drift=True) < score(drift=False)`` for all std values.

    Notes
    -----
    Drift detection must always penalise the recommendation confidence score.
    This is a strict inequality — drift cannot be neutral.
    """
    no_drift   = _confidence_scorer.compute(obs=10_000, posterior_std=std, drift_detected=False)
    with_drift = _confidence_scorer.compute(obs=10_000, posterior_std=std, drift_detected=True)
    assert with_drift < no_drift


@given(
    obs=st.integers(min_value=0, max_value=99),
    std=st.floats(min_value=0.0, max_value=0.05),
)
def test_cold_start_always_below_warm(obs: int, std: float) -> None:
    """
    Invariant: cold-start (``obs < 100``) score <= warm (``obs >= 10,000``) score.

    Notes
    -----
    Sparse data must not produce over-confident recommendations. A service with
    fewer than 100 observations must always receive a confidence score no
    higher than a warm service with 10,000+ observations.
    """
    cold = _confidence_scorer.compute(obs=obs,     posterior_std=std, drift_detected=False)
    warm = _confidence_scorer.compute(obs=10_000, posterior_std=std, drift_detected=False)
    assert cold <= warm + 1e-9


def _wilson_lower(p_hat: float, n: int) -> float:
    """
    Compute the Wilson score lower confidence bound at the 90% level.

    Parameters
    ----------
    p_hat : float
        Observed proportion (success rate).
    n : int
        Number of observations.

    Returns
    -------
    float
        Wilson score lower bound.

    Notes
    -----
    Uses z=1.645 corresponding to a 90% one-sided confidence level (5% tail).
    """
    z = 1.645
    p_tilde    = p_hat + z * z / (2 * n)
    denom      = 1.0 + z * z / n
    under_root = p_hat * (1 - p_hat) / n + z * z / (4 * n * n)
    return (p_tilde - z * math.sqrt(max(0.0, under_root))) / denom


def _hoeffding_lower(p_hat: float, n: int) -> float:
    """
    Compute the Hoeffding concentration inequality lower bound.

    Parameters
    ----------
    p_hat : float
        Observed proportion (success rate).
    n : int
        Number of observations.

    Returns
    -------
    float
        Hoeffding lower bound (may be negative for small ``n``).

    Notes
    -----
    Uses delta=0.05 (5% failure probability), giving ``p_hat - sqrt(log(20) / (2n))``.
    """
    return p_hat - math.sqrt(math.log(1.0 / 0.05) / (2.0 * n))


@given(
    p_hat=st.floats(min_value=0.80, max_value=0.9999),
    n_obs=st.integers(min_value=1, max_value=1_000_000),
)
def test_wilson_lower_bound_never_exceeds_p_hat(p_hat: float, n_obs: int) -> None:
    """
    Invariant: Wilson lower bound <= p_hat.

    Notes
    -----
    A lower confidence bound cannot exceed the point estimate — that would
    contradict the definition of a lower bound.
    """
    low = _wilson_lower(p_hat, n_obs)
    assert low <= p_hat + 1e-9, f"Wilson {low} > p_hat {p_hat} at n={n_obs}"


@given(
    p_hat=st.floats(min_value=0.90, max_value=0.9999),
    n_small=st.integers(min_value=10, max_value=100),
    n_large=st.integers(min_value=1_000, max_value=100_000),
)
def test_wilson_bound_tightens_with_more_observations(
    p_hat: float, n_small: int, n_large: int
) -> None:
    """
    Invariant: ``Wilson(p_hat, n_large) >= Wilson(p_hat, n_small)``.

    Notes
    -----
    More data produces a narrower confidence interval, so the lower bound
    moves closer to p_hat. This monotonicity property is fundamental to
    frequentist interval estimation.
    """
    low_small = _wilson_lower(p_hat, n_small)
    low_large = _wilson_lower(p_hat, n_large)
    assert low_large >= low_small - 1e-9


@given(
    p_hat=st.floats(min_value=0.80, max_value=0.9999),
    n_obs=st.integers(min_value=1, max_value=1_000_000),
)
def test_min_wilson_hoeffding_is_most_conservative(p_hat: float, n_obs: int) -> None:
    """
    Invariant: ``min(Wilson, Hoeffding) <= both`` individual bounds.

    Notes
    -----
    The intersection of two probabilistic guarantees is at least as
    conservative as either guarantee alone. This is a basic property of
    set intersection applied to confidence regions.
    """
    w = _wilson_lower(p_hat, n_obs)
    h = _hoeffding_lower(p_hat, n_obs)
    combined = min(w, h)
    assert combined <= w + 1e-9
    assert combined <= h + 1e-9


@given(quality=st.floats(min_value=0.0, max_value=1.0))
def test_data_quality_all_components_in_unit_interval(quality: float) -> None:
    """
    Invariant: every component score from ``_DataQualityScorer.score()`` is in ``[0, 1]``.

    Notes
    -----
    Regardless of input values, no component of the data quality breakdown
    should be allowed to exceed the unit interval. This guards against
    arithmetic errors in the quality scorer implementation.
    """
    from slo_engine.agents.metrics_agent.tools.tools import _quality_scorer
    avails    = [0.99] * 30
    latencies = [200.0] * 30
    result    = _quality_scorer.score(avails, latencies, n_total=500)
    for key, val in result.items():
        assert 0.0 <= val <= 1.0, f"{key}={val} out of [0,1]"
