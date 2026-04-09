"""
Pydantic schemas for metrics agent tools.

Notes
-----
Defines input and output validation schemas used by the service metrics query,
anomaly detection, and error budget computation tools.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class MetricsQueryInput(BaseModel):
    """
    Input schema for the service metrics query tool.

    Attributes
    ----------
    service_name : str
        Name of the service to query.
    window_days : int
        Observation window in days. Must be between 1 and 90. Defaults to 30.
    include_percentiles : bool
        Whether to include p95 and p99 latency percentiles. Defaults to True.
    include_burn_rates : bool
        Whether to compute 1h, 6h, and 24h error budget burn rates. Defaults to True.

    Notes
    -----
    Used by ``query_service_metrics`` to validate and deserialise the
    JSON-encoded input from ADK tool invocations.
    """

    service_name: str
    window_days: int = Field(default=30, ge=1, le=90)
    include_percentiles: bool = True
    include_burn_rates: bool = True


class MetricsSummary(BaseModel):
    """
    Comprehensive metrics summary produced by the query tool.

    Attributes
    ----------
    service_name : str
        Name of the queried service.
    window_days : int
        Observation window in days.
    mean_availability : float
        Arithmetic mean availability over the window.
    std_availability : float
        Standard deviation of availability.
    p99_latency_ms : float
        99th percentile latency in milliseconds.
    p95_latency_ms : float
        95th percentile latency in milliseconds.
    error_rate_mean : float
        Mean error rate over the window.
    request_count_total : int
        Total number of requests in the window.
    smoothed_availability : float
        Kalman-filtered availability estimate.
    smoothed_p99_ms : float
        Kalman-filtered p99 latency in milliseconds.
    burn_rate_1h : float
        Google SRE 1-hour window error budget burn rate.
    burn_rate_6h : float
        Google SRE 6-hour window error budget burn rate.
    burn_rate_24h : float
        24-hour window error budget burn rate.
    burn_rate_ewma : float
        EWMA-smoothed burn rate with alpha=0.20 over the last 24 hours.
    drift_detected : bool
        Whether KL-divergence drift has been detected.
    kl_divergence : float
        KL-divergence score measuring distribution shift from the baseline.
    posterior_mean : float
        Beta-Binomial posterior mean availability.
    posterior_std : float
        Standard deviation of the Beta-Binomial posterior.
    credible_lower_95 : float
        Lower bound of the 95% Bayesian credible interval.
    credible_upper_95 : float
        Upper bound of the 95% Bayesian credible interval.
    data_quality_score : float
        Composite data quality score (0 to 1).
    data_quality_breakdown : dict
        Per-dimension breakdown of the data quality score.

    Notes
    -----
    ``data_quality_score`` is computed as:
    completeness * 0.4 + staleness * 0.3 + outlier_clean * 0.3.
    ``burn_rate_ewma`` uses exponential weighting with alpha=0.20.
    """

    service_name: str
    window_days: int
    mean_availability: float
    std_availability: float
    p99_latency_ms: float
    p95_latency_ms: float
    error_rate_mean: float
    request_count_total: int
    smoothed_availability: float
    smoothed_p99_ms: float
    burn_rate_1h:  float
    burn_rate_6h:  float
    burn_rate_24h: float
    burn_rate_ewma: float = Field(default=0.0, description="EWMA-smoothed burn rate (alpha=0.20, last 24h)")
    drift_detected: bool
    kl_divergence: float
    posterior_mean: float
    posterior_std: float
    credible_lower_95: float
    credible_upper_95: float
    data_quality_score: float = Field(default=1.0, ge=0.0, le=1.0)
    data_quality_breakdown: dict[str, float] = Field(default_factory=dict)


class AnomalyDetectionInput(BaseModel):
    """
    Input schema for the metric anomaly detection tool.

    Attributes
    ----------
    service_name : str
        Name of the service being evaluated.
    current_availability : float
        Current availability measurement to compare against the baseline.
    baseline_availability : float
        Historical baseline availability for Z-score computation.
    current_p99_ms : float
        Current p99 latency in milliseconds.
    baseline_p99_ms : float
        Historical baseline p99 latency in milliseconds.

    Notes
    -----
    Used by ``detect_metric_anomaly`` to validate input. Z-scores are computed
    as (current - baseline) / std, where std is estimated from the baseline.
    """

    service_name: str
    current_availability: float
    baseline_availability: float
    current_p99_ms: float
    baseline_p99_ms: float


class AnomalyDetectionOutput(BaseModel):
    """
    Output schema for the metric anomaly detection tool.

    Attributes
    ----------
    is_anomaly : bool
        Whether an anomaly was detected based on Z-score thresholds.
    availability_z_score : float
        Z-score of the current availability relative to the baseline.
    latency_z_score : float
        Z-score of the current p99 latency relative to the baseline.
    severity : str
        Severity level string: ``"none"``, ``"low"``, ``"medium"``, ``"high"``,
        or ``"critical"``.
    message : str
        Human-readable description of the anomaly finding.

    Notes
    -----
    Severity thresholds based on the maximum of the two Z-scores:
    >=5 critical, >=4 high, >=3 medium, >=2 low, else none.
    """

    is_anomaly: bool
    availability_z_score: float
    latency_z_score: float
    severity: str
    message: str
