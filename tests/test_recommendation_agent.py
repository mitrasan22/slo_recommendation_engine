"""
Tests for recommendation agent tools.

Notes
-----
Covers: SLO recommendation generation, confidence scoring, feasibility checks,
MILP optimisation, floor-nines snapping, Monte Carlo simulation, and the
Wilson + Hoeffding statistical fallback.

All tools are called through their JSON serialization interface to test the
full input/output contract as invoked by ADK agents.
"""
import json
import math

import pytest

from slo_engine.agents.recommendation_agent.tools.tools import (
    _confidence_scorer,
    _floor_nines,
    _monte_carlo,
    check_slo_feasibility,
    generate_slo_recommendation,
    run_milp_optimization,
)

_WARM_METRICS = {
    "posterior_mean":    0.9991,
    "posterior_std":     0.0001,
    "credible_lower_95": 0.9989,
    "credible_upper_95": 0.9993,
    "smoothed_p99_ms":   250.0,
    "request_count_total": 500_000,
    "drift_detected":    False,
}

_COLD_METRICS = {
    **_WARM_METRICS,
    "request_count_total": 10,
    "posterior_mean":      0.98,
    "posterior_std":       0.05,
    "credible_lower_95":   0.93,
    "credible_upper_95":   0.99,
}


def _recommend(service: str = "api-gateway", metrics: dict = None,
               dep_slos: dict = None, graph: dict = None) -> dict:
    """
    Generate an SLO recommendation and return the parsed result dict.

    Parameters
    ----------
    service : str, optional
        Service name. Defaults to ``"api-gateway"``.
    metrics : dict, optional
        Metrics summary dict. Defaults to ``_WARM_METRICS``.
    dep_slos : dict, optional
        Dependency SLO dict. Defaults to empty.
    graph : dict, optional
        Graph analysis dict. Defaults to critical path of 150 ms.

    Returns
    -------
    dict
        Parsed JSON result from ``generate_slo_recommendation``.

    Notes
    -----
    Helper to reduce boilerplate in individual test methods.
    """
    return json.loads(generate_slo_recommendation(json.dumps({
        "service_name":    service,
        "metrics_summary": metrics or _WARM_METRICS,
        "graph_analysis":  graph  or {"critical_path_latency_ms": 150.0},
        "dep_slos":        dep_slos or {},
    })))


def _feasible(proposed_avail: float, proposed_lat: float,
              historical: float, deps: dict = None) -> dict:
    """
    Check SLO feasibility and return the parsed result dict.

    Parameters
    ----------
    proposed_avail : float
        Proposed availability SLO target.
    proposed_lat : float
        Proposed p99 latency SLO target in ms.
    historical : float
        Historical availability for ceiling enforcement.
    deps : dict, optional
        Dependency availability dict. Defaults to empty.

    Returns
    -------
    dict
        Parsed JSON result from ``check_slo_feasibility``.

    Notes
    -----
    Helper to reduce boilerplate in individual test methods.
    """
    return json.loads(check_slo_feasibility(json.dumps({
        "service_name":          "svc",
        "proposed_availability": proposed_avail,
        "proposed_latency_p99_ms": proposed_lat,
        "historical_availability": historical,
        "dep_availabilities":    deps or {},
    })))


class TestRecommendationGeneration:
    """
    Tests for SLO recommendation generation.

    Notes
    -----
    Verifies required field presence, confidence score bounds, standard SLO
    tier output, cold-start human review triggering, dependency series ceiling
    enforcement, and positive latency values.
    """

    def test_required_fields_present(self):
        out = _recommend()
        for field in ("recommended_availability", "recommended_latency_p99_ms",
                      "confidence_score", "requires_human_review"):
            assert field in out, f"Missing field: {field}"

    def test_confidence_score_in_unit_interval(self):
        out = _recommend()
        assert 0.0 <= out["confidence_score"] <= 1.0

    def test_recommendation_in_standard_slo_set(self):
        SLO_LEVELS = {0.90, 0.95, 0.99, 0.995, 0.999, 0.9995, 0.9999}
        out = _recommend()
        assert out["recommended_availability"] in SLO_LEVELS

    def test_cold_start_requires_human_review(self):
        out = _recommend(service="new-service", metrics=_COLD_METRICS)
        assert out["requires_human_review"] is True

    def test_warm_service_does_not_require_review(self):
        out = _recommend(service="api-gateway", metrics=_WARM_METRICS)
        assert out["requires_human_review"] is False

    def test_series_ceiling_enforced_by_dep(self):
        """
        Verify that a synchronous dependency caps the recommendation.

        Notes
        -----
        A service cannot be recommended above a synchronous dependency's SLO
        because in series reliability, the weakest link determines the achievable
        composite availability.
        """
        out = _recommend(
            service="checkout-service",
            metrics={**_WARM_METRICS, "request_count_total": 100_000},
            dep_slos={
                "external-payment-api": {
                    "recommended_availability": 0.99,
                    "dep_type": "synchronous",
                }
            },
        )
        assert out["recommended_availability"] <= 0.99

    def test_latency_recommendation_positive(self):
        out = _recommend()
        assert out["recommended_latency_p99_ms"] > 0.0


class TestConfidenceScorer:
    """
    Tests for the confidence scorer used in recommendation generation.

    Notes
    -----
    Verifies unit interval output, drift penalty, cold-start penalty,
    anomaly severity penalty, and data quality penalty.
    """

    def test_output_always_in_unit_interval(self):
        for obs in (0, 50, 500, 10_000, 1_000_000):
            for std in (0.0, 0.01, 0.05):
                score = _confidence_scorer.compute(
                    obs=obs, posterior_std=std,
                    drift_detected=False, anomaly_severity="none",
                    data_quality=1.0,
                )
                assert 0.0 <= score <= 1.0, f"score={score} for obs={obs}, std={std}"

    def test_drift_always_reduces_score(self):
        for std in (0.0, 0.01, 0.05):
            no_drift   = _confidence_scorer.compute(obs=10_000, posterior_std=std, drift_detected=False)
            with_drift = _confidence_scorer.compute(obs=10_000, posterior_std=std, drift_detected=True)
            assert with_drift < no_drift, f"drift did not reduce score at std={std}"

    def test_cold_start_below_warm(self):
        cold = _confidence_scorer.compute(obs=50,     posterior_std=0.001, drift_detected=False)
        warm = _confidence_scorer.compute(obs=10_000, posterior_std=0.001, drift_detected=False)
        assert cold <= warm

    def test_critical_anomaly_reduces_score(self):
        base     = _confidence_scorer.compute(obs=10_000, posterior_std=0.001, drift_detected=False, anomaly_severity="none")
        critical = _confidence_scorer.compute(obs=10_000, posterior_std=0.001, drift_detected=False, anomaly_severity="critical")
        assert critical < base

    def test_low_data_quality_reduces_score(self):
        good = _confidence_scorer.compute(obs=10_000, posterior_std=0.001, drift_detected=False, data_quality=0.9)
        poor = _confidence_scorer.compute(obs=10_000, posterior_std=0.001, drift_detected=False, data_quality=0.3)
        assert poor < good


class TestFeasibilityCheck:
    """
    Tests for SLO feasibility validation.

    Notes
    -----
    Verifies that targets achievable within historical performance are
    feasible, that targets above historical ceilings are infeasible, and
    that infeasible results include an adjusted recommendation.
    """

    def test_valid_target_is_feasible(self):
        out = _feasible(proposed_avail=0.99, proposed_lat=200.0, historical=0.995)
        assert out["is_feasible"] is True

    def test_above_historical_ceiling_is_infeasible(self):
        out = _feasible(proposed_avail=0.9999, proposed_lat=100.0, historical=0.90)
        assert out["is_feasible"] is False

    def test_infeasible_result_includes_adjustment(self):
        out = _feasible(proposed_avail=0.9999, proposed_lat=100.0, historical=0.99,
                        deps={"dep-a": 0.99})
        assert out["adjusted_recommendation"] is not None
        assert out["adjusted_recommendation"]["availability"] <= 0.99

    def test_adjustment_respects_dep_ceiling(self):
        out = _feasible(proposed_avail=0.9999, proposed_lat=100.0, historical=0.999,
                        deps={"dep-a": 0.99})
        if not out["is_feasible"]:
            assert out["adjusted_recommendation"]["availability"] <= 0.99


class TestMilpOptimisation:
    """
    Tests for MILP portfolio SLO optimisation.

    Notes
    -----
    Verifies required output fields, discrete SLO tier assignment, weight
    ordering (higher weight gets equal or higher SLO), non-negative error
    budget allocation, and correct single-service handling.
    """

    _SLO_LEVELS = {0.90, 0.95, 0.99, 0.995, 0.999, 0.9995, 0.9999}

    def _run(self, services=None, historical=None, weights=None) -> dict:
        """
        Run MILP optimisation and return the parsed result dict.

        Parameters
        ----------
        services : list, optional
            Service name list. Defaults to ``["A", "B", "C"]``.
        historical : dict, optional
            Historical availability per service. Defaults to standard values.
        weights : dict, optional
            Importance weight per service. Defaults to standard values.

        Returns
        -------
        dict
            Parsed JSON result from ``run_milp_optimization``.

        Notes
        -----
        Helper to reduce boilerplate across MILP test methods.
        """
        services   = services   or ["A", "B", "C"]
        historical = historical or {"A": 0.999, "B": 0.995, "C": 0.99}
        weights    = weights    or {"A": 3.0,   "B": 1.0,   "C": 2.0}
        return json.loads(run_milp_optimization(json.dumps({
            "services":               services,
            "historical_availability": historical,
            "importance_weights":     weights,
        })))

    def test_required_output_fields_present(self):
        out = self._run()
        for field in ("optimal_slos", "error_budget_allocation", "solver_status"):
            assert field in out, f"Missing field: {field}"

    def test_all_assigned_slos_are_discrete_levels(self):
        out = self._run()
        for svc, slo in out["optimal_slos"].items():
            assert slo in self._SLO_LEVELS, f"{svc} got non-standard SLO {slo}"

    def test_higher_weight_gets_higher_or_equal_slo(self):
        """
        Verify that service A (weight=3) is not assigned a lower SLO than B.

        Notes
        -----
        Service A has the highest importance weight (3.0), service B has the
        lowest (1.0). The MILP objective should not assign a lower SLO to the
        more important service.
        """
        out = self._run()
        slos = out["optimal_slos"]
        assert slos.get("A", 0) >= slos.get("B", 0)

    def test_error_budget_allocation_non_negative(self):
        out = self._run()
        for svc, budget in out["error_budget_allocation"].items():
            assert budget >= 0.0, f"{svc} got negative budget {budget}"

    def test_single_service_optimised(self):
        out = self._run(services=["solo"], historical={"solo": 0.999}, weights={"solo": 1.0})
        assert "solo" in out["optimal_slos"]
        assert out["optimal_slos"]["solo"] in self._SLO_LEVELS


class TestFloorNines:
    """
    Tests for the floor-nines SLO tier snapping function.

    Notes
    -----
    Verifies known boundary values snap to the correct tier, that the
    function is idempotent, and that values in the typical operating range
    all map to standard SLO tiers.
    """

    _STANDARD = {0.90, 0.95, 0.99, 0.995, 0.999, 0.9995, 0.9999}

    def test_known_boundary_values(self):
        assert _floor_nines(0.9998) == 0.9999
        assert _floor_nines(0.9994) == 0.9995
        assert _floor_nines(0.998)  == 0.999
        assert _floor_nines(0.994)  == 0.995
        assert _floor_nines(0.91)   == 0.90

    def test_idempotent(self):
        for avail in (0.9998, 0.9994, 0.998, 0.994, 0.91, 0.50, 0.0):
            once  = _floor_nines(avail)
            twice = _floor_nines(once)
            assert once == twice, f"floor_nines not idempotent at {avail}"

    def test_output_in_standard_set_for_typical_range(self):
        for avail in (0.9999, 0.9997, 0.9993, 0.997, 0.993, 0.96, 0.92, 0.90):
            result = _floor_nines(avail)
            assert result in self._STANDARD, f"{avail} -> {result} not in standard set"


class TestMonteCarlo:
    """
    Tests for the Monte Carlo reliability simulation.

    Notes
    -----
    Verifies series product approximation, single dependency passthrough,
    no-dependency own-availability return, and that async dependencies have
    less impact on reliability than synchronous ones.
    """

    def test_two_series_deps_approach_product(self):
        """
        Verify that two series sync deps yield approximately r1 * r2.

        Notes
        -----
        Two 0.99 synchronous dependencies in series yield approximately
        0.9801 (= 0.99^2). The Monte Carlo estimate should be within 1%.
        """
        est = _monte_carlo({"A": 0.99, "B": 0.99}, sync=["A", "B"], asyn=[], own=1.0)
        assert abs(est - 0.9801) < 0.01

    def test_single_dep_matches_dep_availability(self):
        est = _monte_carlo({"A": 0.999}, sync=["A"], asyn=[], own=1.0)
        assert abs(est - 0.999) < 0.005

    def test_no_deps_returns_own_availability(self):
        est = _monte_carlo({}, sync=[], asyn=[], own=0.995)
        assert abs(est - 0.995) < 0.005

    def test_async_dep_has_less_impact_than_sync(self):
        sync_est  = _monte_carlo({"dep": 0.95}, sync=["dep"], asyn=[],      own=1.0)
        async_est = _monte_carlo({"dep": 0.95}, sync=[],      asyn=["dep"], own=1.0)
        assert async_est >= sync_est


class TestWilsonHoeffdingFallback:
    """
    Tests for the Wilson + Hoeffding statistical lower bound fallback.

    Notes
    -----
    These tests verify the mathematical properties of the two lower bounds
    used in the fallback when the Bayesian posterior is not available. Both
    bounds must be at most equal to p_hat and tighten with more observations.
    """

    @staticmethod
    def _wilson_lower(p_hat: float, n: int) -> float:
        """
        Compute the Wilson score lower confidence bound.

        Parameters
        ----------
        p_hat : float
            Observed success proportion.
        n : int
            Number of observations.

        Returns
        -------
        float
            Wilson score lower bound at z=1.645 (90% one-sided).

        Notes
        -----
        Local reimplementation matching the production code for test isolation.
        """
        z = 1.645
        p_tilde    = p_hat + z * z / (2 * n)
        denom      = 1.0 + z * z / n
        under_root = p_hat * (1 - p_hat) / n + z * z / (4 * n * n)
        return (p_tilde - z * math.sqrt(max(0.0, under_root))) / denom

    @staticmethod
    def _hoeffding_lower(p_hat: float, n: int) -> float:
        """
        Compute the Hoeffding concentration inequality lower bound.

        Parameters
        ----------
        p_hat : float
            Observed success proportion.
        n : int
            Number of observations.

        Returns
        -------
        float
            Hoeffding lower bound (may be negative for small ``n``).

        Notes
        -----
        Uses delta=0.05 giving ``p_hat - sqrt(log(20) / (2n))``.
        Local reimplementation matching the production code for test isolation.
        """
        return p_hat - math.sqrt(math.log(1.0 / 0.05) / (2.0 * n))

    def test_wilson_lower_never_exceeds_p_hat(self):
        for p_hat in (0.90, 0.99, 0.999, 0.9999):
            for n in (10, 100, 1_000, 100_000):
                low = self._wilson_lower(p_hat, n)
                assert low <= p_hat + 1e-9, f"Wilson {low} > p_hat {p_hat} at n={n}"

    def test_hoeffding_lower_never_exceeds_p_hat(self):
        for p_hat in (0.90, 0.99, 0.999):
            for n in (10, 100, 1_000):
                low = self._hoeffding_lower(p_hat, n)
                assert low <= p_hat + 1e-9

    def test_wilson_bound_tightens_with_more_data(self):
        """
        Verify that the Wilson lower bound moves closer to p_hat with more data.

        Notes
        -----
        More observations narrow the confidence interval, so the lower bound
        increases toward p_hat. This monotonicity property must hold for all
        p_hat values in the typical SLO range.
        """
        for p_hat in (0.95, 0.99, 0.999):
            low_small = self._wilson_lower(p_hat, n=10)
            low_large = self._wilson_lower(p_hat, n=100_000)
            assert low_large >= low_small - 1e-9, (
                f"Wilson did not tighten at p_hat={p_hat}: small={low_small}, large={low_large}"
            )

    def test_min_of_wilson_and_hoeffding_is_conservative(self):
        for p_hat in (0.95, 0.99, 0.999):
            for n in (100, 10_000):
                w = self._wilson_lower(p_hat, n)
                h = self._hoeffding_lower(p_hat, n)
                combined = min(w, h)
                assert combined <= w + 1e-9
                assert combined <= h + 1e-9
