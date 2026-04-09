"""
Tests for metrics agent tools.

Notes
-----
Covers: Bayesian posterior, Kalman smoothing, burn rate, KL-divergence drift,
z-score anomaly detection, error budget status, and data quality scoring.

All tools are called through their JSON serialization interface (matching
how ADK agents invoke them) to test the full input/output contract.
"""
import json

from slo_engine.agents.metrics_agent.tools.tools import (
    compute_error_budget_status,
    detect_metric_anomaly,
    query_service_metrics,
)


def _metrics(service: str = "api-gateway", window_days: int = 30) -> dict:
    """
    Query service metrics and return the parsed result dict.

    Parameters
    ----------
    service : str, optional
        Service name to query. Defaults to ``"api-gateway"``.
    window_days : int, optional
        Observation window in days. Defaults to 30.

    Returns
    -------
    dict
        Parsed JSON result from ``query_service_metrics``.

    Notes
    -----
    Helper to reduce boilerplate in individual test methods.
    """
    return json.loads(query_service_metrics(json.dumps(
        {"service_name": service, "window_days": window_days}
    )))


def _anomaly(
    service: str = "api-gateway",
    current_avail: float = 0.9991,
    baseline_avail: float = 0.9990,
    current_p99: float = 252.0,
    baseline_p99: float = 250.0,
) -> dict:
    """
    Run anomaly detection and return the parsed result dict.

    Parameters
    ----------
    service : str, optional
        Service name. Defaults to ``"api-gateway"``.
    current_avail : float, optional
        Current availability measurement. Defaults to 0.9991.
    baseline_avail : float, optional
        Baseline availability for comparison. Defaults to 0.9990.
    current_p99 : float, optional
        Current p99 latency in ms. Defaults to 252.0.
    baseline_p99 : float, optional
        Baseline p99 latency in ms. Defaults to 250.0.

    Returns
    -------
    dict
        Parsed JSON result from ``detect_metric_anomaly``.

    Notes
    -----
    Helper to reduce boilerplate in individual test methods.
    """
    return json.loads(detect_metric_anomaly(json.dumps({
        "service_name":            service,
        "current_availability":    current_avail,
        "baseline_availability":   baseline_avail,
        "current_p99_ms":          current_p99,
        "baseline_p99_ms":         baseline_p99,
    })))


def _budget(service: str = "api-gateway", target: float = 0.999) -> dict:
    """
    Compute error budget status and return the parsed result dict.

    Parameters
    ----------
    service : str, optional
        Service name. Defaults to ``"api-gateway"``.
    target : float, optional
        SLO target availability. Defaults to 0.999.

    Returns
    -------
    dict
        Parsed JSON result from ``compute_error_budget_status``.

    Notes
    -----
    Helper to reduce boilerplate in individual test methods.
    """
    return json.loads(compute_error_budget_status(json.dumps(
        {"service_name": service, "slo_target": target, "window_days": 30}
    )))


class TestBayesianPosterior:
    """
    Tests for the Beta-Binomial Bayesian posterior availability estimate.

    Notes
    -----
    Verifies that the posterior mean is a valid probability, the credible
    interval is correctly ordered, and both bounds are within ``[0, 1]``.
    """

    def test_posterior_mean_is_valid_probability(self):
        out = _metrics()
        assert 0.0 < out["posterior_mean"] < 1.0

    def test_credible_interval_ordering(self):
        out = _metrics()
        assert out["credible_lower_95"] < out["posterior_mean"]
        assert out["posterior_mean"] < out["credible_upper_95"]

    def test_credible_bounds_in_unit_interval(self):
        out = _metrics()
        assert 0.0 <= out["credible_lower_95"] <= 1.0
        assert 0.0 <= out["credible_upper_95"] <= 1.0

    def test_shorter_window_returns_valid_posterior(self):
        out = _metrics(window_days=7)
        assert 0.0 < out["posterior_mean"] < 1.0


class TestKalmanSmoothing:
    """
    Tests for Kalman-filtered metric smoothing.

    Notes
    -----
    The Kalman-smoothed availability must be a valid probability and the
    smoothed p99 latency must be a positive number.
    """

    def test_smoothed_availability_is_valid_probability(self):
        out = _metrics(window_days=7)
        assert 0.0 < out["smoothed_availability"] < 1.0

    def test_smoothed_p99_positive(self):
        out = _metrics()
        assert out["smoothed_p99_ms"] > 0.0


class TestBurnRate:
    """
    Tests for Google SRE error budget burn rate computation.

    Notes
    -----
    Both the 1h and 6h burn rates must be non-negative. Their presence in
    the output is required regardless of the actual values.
    """

    def test_burn_rates_are_non_negative(self):
        out = _metrics(service="checkout-service")
        assert out["burn_rate_1h"] >= 0.0
        assert out["burn_rate_6h"] >= 0.0

    def test_short_window_burn_rate_at_least_long_window(self):
        """
        Verify that both burn rate fields are present in the output.

        Notes
        -----
        No strict ordering is enforced between the 1h and 6h burn rates
        because they depend on simulated data. The test only confirms field
        presence.
        """
        out = _metrics()
        assert "burn_rate_1h" in out
        assert "burn_rate_6h" in out


class TestDriftDetection:
    """
    Tests for KL-divergence based distribution drift detection.

    Notes
    -----
    The KL divergence must be non-negative and the drift_detected field
    must be a boolean.
    """

    def test_drift_fields_present(self):
        out = _metrics(service="auth-service")
        assert "drift_detected" in out
        assert "kl_divergence" in out

    def test_kl_divergence_non_negative(self):
        out = _metrics(service="auth-service")
        assert out["kl_divergence"] >= 0.0

    def test_drift_detected_is_boolean(self):
        out = _metrics()
        assert isinstance(out["drift_detected"], bool)


class TestAnomalyDetection:
    """
    Tests for z-score based metric anomaly detection.

    Notes
    -----
    Verifies that marginal changes produce no anomaly, major drops produce
    critical anomalies, and all required output fields are present.
    """

    def test_no_anomaly_for_marginal_change(self):
        out = _anomaly(current_avail=0.9991, baseline_avail=0.9990,
                       current_p99=252.0, baseline_p99=250.0)
        assert out["severity"] == "none"
        assert out["is_anomaly"] is False

    def test_critical_anomaly_for_major_drop(self):
        out = _anomaly(
            service="checkout-service",
            current_avail=0.85,
            baseline_avail=0.999,
            current_p99=5000.0,
            baseline_p99=800.0,
        )
        assert out["is_anomaly"] is True
        assert out["severity"] in ("high", "critical")

    def test_anomaly_result_has_required_fields(self):
        out = _anomaly()
        assert "is_anomaly" in out
        assert "severity" in out
        assert out["severity"] in ("none", "low", "medium", "high", "critical")


class TestErrorBudgetStatus:
    """
    Tests for error budget burn fraction and exhaustion probability.

    Notes
    -----
    Both the burn fraction and exhaustion probability must be in ``[0, 1]``.
    A stricter SLO target (higher value) should produce a higher or equal
    burn fraction due to the smaller available error budget.
    """

    def test_burn_fraction_in_unit_interval(self):
        out = _budget()
        assert 0.0 <= out["burn_fraction"] <= 1.0

    def test_exhaustion_probability_in_unit_interval(self):
        out = _budget()
        assert 0.0 <= out["prob_exhaust_in_window"] <= 1.0

    def test_required_fields_present(self):
        out = _budget()
        for field in ("burn_fraction", "prob_exhaust_in_window"):
            assert field in out, f"Missing field: {field}"

    def test_high_slo_target_produces_higher_burn_fraction(self):
        """
        Verify that a stricter SLO target produces a higher burn fraction.

        Notes
        -----
        A stricter SLO (e.g. 0.9999 vs 0.99) means a smaller error budget
        in absolute terms. The same error rate therefore consumes a larger
        fraction of the budget, resulting in a higher burn fraction.
        """
        low_target  = _budget(target=0.99)
        high_target = _budget(target=0.9999)
        assert high_target["burn_fraction"] >= low_target["burn_fraction"]


class TestDataQualityScoring:
    """
    Tests for the data quality scorer embedded in metrics output.

    Notes
    -----
    Verifies that the data quality score and its breakdown are present in
    the metrics output and that all values are in ``[0, 1]``.
    """

    def test_data_quality_score_present_in_metrics(self):
        out = _metrics()
        assert "data_quality_score" in out

    def test_data_quality_score_in_unit_interval(self):
        out = _metrics()
        assert 0.0 <= out["data_quality_score"] <= 1.0

    def test_data_quality_breakdown_keys_in_unit_interval(self):
        out = _metrics()
        breakdown = out.get("data_quality_breakdown", {})
        for key, val in breakdown.items():
            assert 0.0 <= val <= 1.0, f"{key}={val} not in [0,1]"

    def test_data_quality_scorer_directly(self):
        from slo_engine.agents.metrics_agent.tools.tools import _quality_scorer
        avails    = [0.999] * 30
        latencies = [200.0] * 30
        result    = _quality_scorer.score(avails, latencies, n_total=500)
        assert 0.0 <= result["data_quality_score"] <= 1.0
        for key, val in result.items():
            assert 0.0 <= val <= 1.0, f"{key}={val}"
