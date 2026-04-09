"""
Metrics agent tools implementing Bayesian inference and time-series math.

Notes
-----
Bayesian inference, Kalman filtering, burn-rate math, and data quality
scoring live directly in these tool functions — registered as FunctionTool
with Google ADK.

Mathematics used:
  - Beta-Binomial conjugate update: p|data ~ Beta(alpha+k, beta+n-k)
  - Kalman filter (1-D random walk): K = P/(P+R), x = x + K(z-x)
  - KL divergence (drift detection): KL(P||Q) = sum(P * log(P/Q))
  - Google SRE burn rate: burn = error_rate / (1-SLO)
  - Gaussian error-budget forecast: P(exhaust) = Phi((E-forecast)/sigma)
  - Z-score anomaly detection: z = |x - mu| / sigma
  - Data quality score = 0.4*completeness + 0.3*staleness + 0.3*outlier
"""
from __future__ import annotations

import json
import math
import random
from typing import ClassVar

import numpy as np
from scipy import stats

from slo_engine.agents.metrics_agent.tools.schema import (
    AnomalyDetectionInput,
    AnomalyDetectionOutput,
    MetricsQueryInput,
    MetricsSummary,
)
from slo_engine.config.settings import settings

_PRIOR_ALPHA: float = float(settings.compute_logic.bayesian_prior_alpha)
_PRIOR_BETA:  float = float(settings.compute_logic.bayesian_prior_beta)

_KALMAN_Q: float = float(settings.compute_logic.kalman_process_noise)
_KALMAN_R: float = float(settings.compute_logic.kalman_measurement_noise)

_EWMA_ALPHA: float = 0.20


class _DataQualityScorer:
    """
    Score ingested metrics on completeness, staleness, and outlier cleanliness.

    Attributes
    ----------
    COMPLETENESS_W : float
        Weight for the completeness dimension (0.40).
    STALENESS_W : float
        Weight for the staleness dimension (0.30).
    OUTLIER_W : float
        Weight for the outlier-cleanliness dimension (0.30).
    SIGMA_THRESH : float
        Number of standard deviations used as the outlier threshold (3.0).
    MIN_REQUESTS : int
        Minimum total request count for full completeness score (100).
    MIN_OBS_HOURS : int
        Minimum observation count for full staleness score (24).

    Notes
    -----
    The final composite score (0 to 1) feeds into the confidence scorer.
    A quality score below 0.6 incurs a confidence penalty regardless of the
    Bayesian posterior width.
    """

    COMPLETENESS_W: ClassVar[float] = 0.40
    STALENESS_W:    ClassVar[float] = 0.30
    OUTLIER_W:      ClassVar[float] = 0.30
    SIGMA_THRESH:   ClassVar[float] = 3.0
    MIN_REQUESTS:   ClassVar[int]   = 100
    MIN_OBS_HOURS:  ClassVar[int]   = 24

    def score(
        self,
        avails:    list[float],
        latencies: list[float],
        n_total:   int,
    ) -> dict[str, float]:
        """
        Compute the composite quality score with per-component breakdown.

        Parameters
        ----------
        avails : list of float
            Time-series of availability measurements.
        latencies : list of float
            Time-series of latency measurements in milliseconds.
        n_total : int
            Total request count in the observation window.

        Returns
        -------
        dict of str to float
            Dictionary with keys ``data_quality_score``, ``completeness_score``,
            ``staleness_score``, and ``outlier_score``.

        Notes
        -----
        Composite score: completeness * 0.40 + staleness * 0.30 + outlier * 0.30.
        """
        completeness = self._completeness(avails, latencies, n_total)
        staleness    = self._staleness(avails)
        outlier      = self._outlier_score(avails)
        total = (
            completeness * self.COMPLETENESS_W
            + staleness  * self.STALENESS_W
            + outlier    * self.OUTLIER_W
        )
        return {
            "data_quality_score": round(total, 4),
            "completeness_score": round(completeness, 4),
            "staleness_score":    round(staleness, 4),
            "outlier_score":      round(outlier, 4),
        }

    def _completeness(
        self, avails: list[float], latencies: list[float], n_total: int
    ) -> float:
        """
        Score completeness based on presence of both time series and request volume.

        Parameters
        ----------
        avails : list of float
            Availability time series.
        latencies : list of float
            Latency time series.
        n_total : int
            Total request count.

        Returns
        -------
        float
            Score between 0.0 and 1.0. Each missing condition (no availability
            series, no latency series, insufficient volume) costs 1/3 of the score.

        Notes
        -----
        Full score (1.0) requires all three conditions to be satisfied.
        """
        has_avail    = len(avails) > 0
        has_latency  = len(latencies) > 0
        has_volume   = n_total >= self.MIN_REQUESTS
        return sum([has_avail, has_latency, has_volume]) / 3.0

    def _staleness(self, avails: list[float]) -> float:
        """
        Score data freshness via observation count as a proxy.

        Parameters
        ----------
        avails : list of float
            Availability time series. Length is used as observation count.

        Returns
        -------
        float
            Score between 0.0 and 1.0. Full score at >= MIN_OBS_HOURS observations;
            linearly decays to 0 below.

        Notes
        -----
        In production this should be replaced with a timestamp-based staleness check
        comparing ``now - last_seen_timestamp`` against a freshness threshold.
        """
        return min(1.0, len(avails) / self.MIN_OBS_HOURS)

    def _outlier_score(self, avails: list[float]) -> float:
        """
        Score based on 3-sigma outlier contamination fraction.

        Parameters
        ----------
        avails : list of float
            Availability time series.

        Returns
        -------
        float
            Score between 0.0 and 1.0. Score degrades as outlier fraction rises
            above 5%; returns 0.5 when fewer than 4 observations are available.

        Notes
        -----
        Score degrades linearly: 0% outliers -> 1.0, 5% -> 1.0, 10% -> 0.5,
        15%+ -> 0.0. The 5% tolerance accommodates minor data pipeline noise.
        """
        n = len(avails)
        if n < 4:
            return 0.5

        arr   = np.array(avails)
        mu    = arr.mean()
        sigma = arr.std()
        if sigma < 1e-10:
            return 1.0

        outlier_frac = float(np.mean(np.abs(arr - mu) > self.SIGMA_THRESH * sigma))
        excess = max(0.0, outlier_frac - 0.05)
        return max(0.0, 1.0 - excess * 10.0)


_quality_scorer = _DataQualityScorer()

_MOCK: dict[str, dict] = {
    "api-gateway":       {"avail": [0.9991 + random.gauss(0, 3e-4) for _ in range(720)],
                          "p99":   [250.0  + random.gauss(0, 30)   for _ in range(720)],
                          "req":   [1000   + random.randint(-100, 100) for _ in range(720)],
                          "err":   [random.randint(0, 2)            for _ in range(720)]},
    "checkout-service":  {"avail": [0.995  + random.gauss(0, 1e-3) for _ in range(720)],
                          "p99":   [800.0  + random.gauss(0, 100)  for _ in range(720)],
                          "req":   [500    + random.randint(-50,50) for _ in range(720)],
                          "err":   [random.randint(2, 8)            for _ in range(720)]},
    "auth-service":      {"avail": [0.9998 + random.gauss(0, 1e-4) for _ in range(720)],
                          "p99":   [50.0   + random.gauss(0, 10)   for _ in range(720)],
                          "req":   [3000   + random.randint(-200, 200) for _ in range(720)],
                          "err":   [random.randint(0, 1)            for _ in range(720)]},
}


def query_service_metrics(
    query_json: str = "",
    service_name: str = "",
    window_days: int = 30,
    include_percentiles: bool = True,
    include_burn_rates: bool = True,
) -> str:
    """
    Query historical metrics and compute Bayesian posterior, Kalman-smoothed
    availability, burn rates, and KL-divergence drift detection.

    Parameters
    ----------
    query_json : str, optional
        JSON-encoded ``MetricsQueryInput`` object. When provided, individual
        keyword arguments are ignored.
    service_name : str, optional
        Name of the service to query. Used when ``query_json`` is empty.
    window_days : int, optional
        Observation window in days. Defaults to 30.
    include_percentiles : bool, optional
        Whether to include p95 and p99 latency percentiles. Defaults to True.
    include_burn_rates : bool, optional
        Whether to compute burn rates. Defaults to True.

    Returns
    -------
    str
        JSON-encoded ``MetricsSummary`` on success, or JSON with
        ``status="error"`` and a ``message`` field on failure.

    Notes
    -----
    Beta-Binomial conjugate update: posterior ~ Beta(alpha+k, beta+n-k) where
    k = successes, n = total requests. Kalman filter uses 1-D random walk:
    K = P/(P+R), x = x + K*(z-x). Burn rates use the Google SRE formula:
    burn = error_rate / (1 - SLO). KL-divergence is computed between the first
    and second halves of the observation window to detect distributional drift.
    EWMA burn rate uses alpha=0.20 over the last 24 hours.
    """
    try:
        if query_json:
            inp = MetricsQueryInput.model_validate_json(query_json)
        else:
            inp = MetricsQueryInput(
                service_name=service_name,
                window_days=window_days,
                include_percentiles=include_percentiles,
                include_burn_rates=include_burn_rates,
            )
        svc = inp.service_name
        window_h = inp.window_days * 24

        raw = _MOCK.get(svc) or {
            "avail": [0.99 + random.gauss(0, 2e-3) for _ in range(window_h)],
            "p99":   [200.0 + random.gauss(0, 50)  for _ in range(window_h)],
            "req":   [500] * window_h,
            "err":   [5]   * window_h,
        }

        avails  = [max(0.0, min(1.0, a)) for a in raw["avail"][-window_h:]]
        latencies = [max(1.0, l)          for l in raw["p99"][-window_h:]]
        requests = raw["req"][-window_h:]
        errors   = raw["err"][-window_h:]

        n_total  = sum(requests)
        n_errors = sum(errors)
        k        = n_total - n_errors

        alpha_post = _PRIOR_ALPHA + k
        beta_post  = _PRIOR_BETA  + n_errors

        post_dist  = stats.beta(alpha_post, beta_post)
        post_mean  = alpha_post / (alpha_post + beta_post)
        post_std   = math.sqrt(alpha_post * beta_post /
                               ((alpha_post + beta_post)**2 * (alpha_post + beta_post + 1)))
        ci_95 = post_dist.interval(0.95)

        raw_slo = float(ci_95[0])
        rec_slo = _floor_nines(max(0.90, raw_slo))

        x_hat, P = avails[0], _KALMAN_R
        smoothed = []
        for z in avails:
            P_minus = P + _KALMAN_Q
            K = P_minus / (P_minus + _KALMAN_R)
            x_hat = x_hat + K * (z - x_hat)
            P = (1.0 - K) * P_minus
            smoothed.append(float(np.clip(x_hat, 0.0, 1.0)))
        kalman_avail = smoothed[-1]

        error_budget_frac = max(1e-10, 1.0 - rec_slo)
        burn_rates: dict[str, float] = {}
        for w_h in [1, 6, 24]:
            tail_e = sum(errors[-w_h:])
            tail_r = sum(requests[-w_h:])
            er = tail_e / max(tail_r, 1)
            burn_rates[f"{w_h}h"] = round(er / error_budget_frac, 4)

        hourly_er = [
            errors[i] / max(requests[i], 1)
            for i in range(-min(24, len(errors)), 0)
        ]
        ewma_er = hourly_er[0] if hourly_er else 0.0
        for er_h in hourly_er[1:]:
            ewma_er = _EWMA_ALPHA * er_h + (1.0 - _EWMA_ALPHA) * ewma_er
        burn_rates["ewma"] = round(ewma_er / error_budget_frac, 4)

        half = len(avails) // 2
        baseline, current = avails[:half], avails[half:]
        kl = 0.0
        drift = False
        if half >= 10:
            bins = np.linspace(min(avails), max(avails) + 1e-9, 51)
            eps  = 1e-10
            p_h, _ = np.histogram(current,  bins=bins, density=True)
            q_h, _ = np.histogram(baseline, bins=bins, density=True)
            p = (p_h + eps) / (p_h + eps).sum()
            q = (q_h + eps) / (q_h + eps).sum()
            kl = float(np.sum(p * np.log(p / q)))
            drift = kl > 0.1

        lat_arr = np.array(latencies)

        quality = _quality_scorer.score(avails, latencies, n_total)

        summary = MetricsSummary(
            service_name=svc,
            window_days=inp.window_days,
            mean_availability=float(np.mean(avails)),
            std_availability=float(np.std(avails)),
            p99_latency_ms=float(np.percentile(lat_arr, 99)),
            p95_latency_ms=float(np.percentile(lat_arr, 95)),
            error_rate_mean=float(np.mean([e / max(r, 1) for e, r in zip(errors, requests)])),
            request_count_total=n_total,
            smoothed_availability=kalman_avail,
            smoothed_p99_ms=float(np.mean(lat_arr)),
            burn_rate_1h=burn_rates.get("1h", 0.0),
            burn_rate_6h=burn_rates.get("6h", 0.0),
            burn_rate_24h=burn_rates.get("24h", 0.0),
            burn_rate_ewma=burn_rates.get("ewma", 0.0),
            drift_detected=drift,
            kl_divergence=round(kl, 4),
            posterior_mean=post_mean,
            posterior_std=post_std,
            credible_lower_95=float(ci_95[0]),
            credible_upper_95=float(ci_95[1]),
            data_quality_score=quality["data_quality_score"],
            data_quality_breakdown={
                k: v for k, v in quality.items() if k != "data_quality_score"
            },
        )
        return summary.model_dump_json()
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


def detect_metric_anomaly(
    anomaly_input_json: str = "",
    service_name: str = "",
    current_availability: float = 0.99,
    baseline_availability: float = 0.99,
    current_p99_ms: float = 200.0,
    baseline_p99_ms: float = 200.0,
) -> str:
    """
    Detect metric anomalies using Z-score comparison to baseline.

    Parameters
    ----------
    anomaly_input_json : str, optional
        JSON-encoded ``AnomalyDetectionInput`` object. When provided, individual
        keyword arguments are ignored.
    service_name : str, optional
        Name of the service being evaluated.
    current_availability : float, optional
        Current availability measurement. Defaults to 0.99.
    baseline_availability : float, optional
        Historical baseline availability. Defaults to 0.99.
    current_p99_ms : float, optional
        Current p99 latency in milliseconds. Defaults to 200.0.
    baseline_p99_ms : float, optional
        Historical baseline p99 latency in milliseconds. Defaults to 200.0.

    Returns
    -------
    str
        JSON-encoded ``AnomalyDetectionOutput`` on success, or JSON with
        ``status="error"`` and a ``message`` field on failure.

    Notes
    -----
    Z-score formula: z = |x - mu| / sigma, where sigma is estimated as
    1% of baseline availability and 10% of baseline latency. Severity
    thresholds: z >= 5 critical, >= 4 high, >= 3 medium, >= 2 low, else none.
    A Z-score > 2 represents a 2-sigma event (p < 5%).
    A Z-score > 3 represents a 3-sigma event (p < 0.3%).
    """
    try:
        if anomaly_input_json:
            inp = AnomalyDetectionInput.model_validate_json(anomaly_input_json)
        else:
            inp = AnomalyDetectionInput(
                service_name=service_name,
                current_availability=current_availability,
                baseline_availability=baseline_availability,
                current_p99_ms=current_p99_ms,
                baseline_p99_ms=baseline_p99_ms,
            )

        avail_sigma = max(1e-6, inp.baseline_availability * 0.01)
        lat_sigma   = max(1.0,  inp.baseline_p99_ms * 0.10)

        avail_z = abs(inp.current_availability - inp.baseline_availability) / avail_sigma
        lat_z   = abs(inp.current_p99_ms - inp.baseline_p99_ms) / lat_sigma
        max_z   = max(avail_z, lat_z)

        severity = (
            "critical" if max_z >= 5.0 else
            "high"     if max_z >= 4.0 else
            "medium"   if max_z >= 3.0 else
            "low"      if max_z >= 2.0 else
            "none"
        )
        out = AnomalyDetectionOutput(
            is_anomaly=max_z >= 2.0,
            availability_z_score=round(avail_z, 3),
            latency_z_score=round(lat_z, 3),
            severity=severity,
            message=f"avail z={avail_z:.2f}, latency z={lat_z:.2f} -> {severity}",
        )
        return out.model_dump_json()
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


def compute_error_budget_status(
    budget_input_json: str = "",
    service_name: str = "",
    slo_target: float = 0.999,
    window_days: int = 30,
) -> str:
    """
    Compute error budget health and Gaussian probability of budget exhaustion.

    Parameters
    ----------
    budget_input_json : str, optional
        JSON-encoded object with ``service_name``, ``slo_target``, and
        ``window_days``. When provided, individual keyword arguments are ignored.
    service_name : str, optional
        Name of the service to compute the budget for.
    slo_target : float, optional
        Target SLO availability between 0 and 1. Defaults to 0.999.
    window_days : int, optional
        Budget window in days. Defaults to 30.

    Returns
    -------
    str
        JSON object with ``burn_fraction``, ``burn_rate_per_day``,
        ``days_to_exhaustion``, ``prob_exhaust_in_window``, and ``status`` fields.
        Returns JSON with ``status="error"`` on failure.

    Notes
    -----
    Error budget for window: E = (1-SLO) * window_hours.
    Consumed budget: C = sum(1 - avail_i) over all observed hours.
    Burn rate: r = C / n_obs (per hour).
    Forecast: C_final ~ N(C + H_rem*r, H_rem*sigma_r^2).
    P(exhaust) = 1 - Phi((E - forecast_mean) / forecast_std).
    Status thresholds: burn_fraction > 0.9 -> critical, > 0.5 -> warning.
    """
    try:
        if budget_input_json:
            inp = json.loads(budget_input_json)
            svc    = inp["service_name"]
            slo    = float(inp.get("slo_target", 0.999))
            window = int(inp.get("window_days", 30))
        else:
            svc    = service_name
            slo    = slo_target
            window = window_days

        raw    = _MOCK.get(svc, {})
        avails = raw.get("avail", [0.999] * 720)
        window_h = window * 24
        avails   = [max(0.0, min(1.0, a)) for a in avails[-window_h:]]

        budget_hours = (1.0 - slo) * window_h
        errors_h     = [1.0 - a for a in avails]
        consumed     = float(np.sum(errors_h))
        n_obs        = len(errors_h)
        burn_rate    = consumed / n_obs if n_obs > 0 else 0.0
        burn_var     = float(np.var(errors_h)) if n_obs > 1 else 0.0

        remaining_h  = max(0.0, window_h - n_obs)
        days_to_exhaust = (
            (budget_hours - consumed) / burn_rate / 24
            if burn_rate > 0 else float("inf")
        )

        forecast_mean = consumed + remaining_h * burn_rate
        forecast_var  = remaining_h * burn_var
        forecast_std  = math.sqrt(forecast_var) if forecast_var > 0 else 1e-9

        z = (budget_hours - forecast_mean) / forecast_std
        prob_exhaust = float(1.0 - stats.norm.cdf(z))
        burn_frac    = min(1.0, consumed / max(budget_hours, 1e-9))

        return json.dumps({
            "service": svc,
            "slo_target": slo,
            "burn_fraction": round(burn_frac, 4),
            "burn_rate_per_day": round(burn_rate * 24, 6),
            "days_to_exhaustion": round(max(0.0, days_to_exhaust), 1),
            "prob_exhaust_in_window": round(prob_exhaust, 4),
            "status": (
                "critical" if burn_frac > 0.9 else
                "warning"  if burn_frac > 0.5 else
                "healthy"
            ),
        })
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


def _floor_nines(v: float) -> float:
    """
    Floor a value to the nearest standard SLO boundary.

    Parameters
    ----------
    v : float
        Raw SLO candidate value between 0 and 1.

    Returns
    -------
    float
        Largest standard SLO boundary not exceeding ``v``, or ``v`` rounded
        to 4 decimal places if below 0.90.

    Notes
    -----
    Standard SLO boundaries checked in descending order:
    0.9999, 0.9995, 0.999, 0.995, 0.99, 0.95, 0.90.
    Used to snap the Bayesian lower credible bound to a conventional SLO tier.
    """
    for b in [0.9999, 0.9995, 0.999, 0.995, 0.99, 0.95, 0.90]:
        if v >= b:
            return b
    return round(v, 4)
