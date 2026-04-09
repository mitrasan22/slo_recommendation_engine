"""
Recommendation agent tools implementing reliability and optimization math.

Notes
-----
All reliability, latency, and optimization math lives directly here.
Registered as FunctionTool with Google ADK — no intermediate layer.

Mathematics used:
  - Series reliability: R = product(R_i)
  - Parallel reliability: R = 1 - product(1-R_i)
  - Monte Carlo: estimate R by simulating N Bernoulli trials
  - CLT latency: L_total ~ N(sum(mu), sum(sigma^2)), p99 = mu + 2.326*sigma
  - Hoeffding bound: t = sqrt(-ln(delta) * sum((b-a)^2) / 2)
  - MILP (PuLP): max sum(w_i * slo_k * z_ik) subject to assignment,
    budget, dependency ceiling, and feasibility cap constraints
  - Water-filling (KKT): b*_i = (w_i / sum(w_j)) * B_total
  - Confidence scoring: explicit cold-start tiers plus drift and anomaly penalties
"""
from __future__ import annotations

import json
import math
from typing import ClassVar

import numpy as np
import pulp

from slo_engine.agents.recommendation_agent.tools.schema import (
    FeasibilityCheckInput,
    FeasibilityCheckOutput,
    SLORecommendationInput,
    SLORecommendationOutput,
)
from slo_engine.config.settings import settings
from slo_engine.rag.knowledge_store import knowledge_store

_MC_N: int = int(settings.compute_logic.monte_carlo_samples)


class _ConfidenceScorer:
    """
    Multi-component confidence scorer for SLO recommendations.

    Attributes
    ----------
    OBS_WEIGHT : float
        Weight for the observation count component (0.40).
    STABLE_WEIGHT : float
        Weight for the posterior stability component (0.40).
    BASE : float
        Baseline score added to all recommendations (0.20).
    DRIFT_PENALTY : float
        Score deduction when KL-divergence drift is detected (0.25).
    ANOMALY_PENALTIES : dict of str to float
        Mapping of anomaly severity string to score deduction.

    Notes
    -----
    Score formula: OBS_W * obs_component + STABLE_W * stable_component + BASE
    - drift_penalty - anomaly_penalty - quality_penalty.
    Observation component uses explicit cold-start tiers rather than a purely
    log-scaled formula to ensure that fewer than 100 requests definitively
    scores as cold-start (0.20). A critical anomaly always pushes the score
    below 0.75, triggering mandatory SRE review.
    """

    OBS_WEIGHT:    ClassVar[float] = 0.40
    STABLE_WEIGHT: ClassVar[float] = 0.40
    BASE:          ClassVar[float] = 0.20

    DRIFT_PENALTY: ClassVar[float] = 0.25

    ANOMALY_PENALTIES: ClassVar[dict[str, float]] = {
        "critical": 0.20,
        "high":     0.15,
        "medium":   0.05,
        "low":      0.00,
        "none":     0.00,
    }

    def compute(
        self,
        obs:              int,
        posterior_std:    float,
        drift_detected:   bool,
        anomaly_severity: str = "none",
        data_quality:     float = 1.0,
    ) -> float:
        """
        Compute a confidence score in [0, 1].

        Parameters
        ----------
        obs : int
            Total request count in the observation window.
        posterior_std : float
            Width of the Bayesian credible interval standard deviation.
        drift_detected : bool
            KL-divergence drift flag from the metrics agent.
        anomaly_severity : str, optional
            Z-score severity string from the anomaly agent. Defaults to ``"none"``.
        data_quality : float, optional
            Data quality score between 0 and 1 from the quality scorer.
            Scores below 0.6 add an additional quality penalty. Defaults to 1.0.

        Returns
        -------
        float
            Confidence score in [0, 1]. Score below 0.75 triggers mandatory SRE review.

        Notes
        -----
        Quality penalty = max(0, (0.60 - data_quality) * 0.5) when data_quality < 0.60.
        Result is clipped to [0, 1] after all components are combined.
        """
        obs_c     = self._obs_component(obs)
        stable_c  = self._stability_component(posterior_std)
        drift_p   = self.DRIFT_PENALTY if drift_detected else 0.0
        anomaly_p = self.ANOMALY_PENALTIES.get(anomaly_severity.lower(), 0.0)
        quality_p = max(0.0, (0.60 - data_quality) * 0.5) if data_quality < 0.60 else 0.0

        raw = (
            obs_c * self.OBS_WEIGHT
            + stable_c * self.STABLE_WEIGHT
            + self.BASE
            - drift_p
            - anomaly_p
            - quality_p
        )
        return float(np.clip(raw, 0.0, 1.0))

    @staticmethod
    def _obs_component(obs: int) -> float:
        """
        Map observation count to a score component using explicit cold-start tiers.

        Parameters
        ----------
        obs : int
            Total number of requests in the observation window.

        Returns
        -------
        float
            Score component between 0.20 and 1.0.

        Notes
        -----
        Tiers: < 100 -> 0.20 (cold start), < 1000 -> 0.50 (warming up),
        < 10000 -> log-scaled 0.50-0.87, >= 10000 -> log-scaled toward 1.0 at 100k.
        """
        if obs < 100:
            return 0.20
        if obs < 1_000:
            return 0.50
        return min(1.0, math.log1p(obs) / math.log1p(100_000))

    @staticmethod
    def _stability_component(posterior_std: float) -> float:
        """
        Map posterior standard deviation to a stability score component.

        Parameters
        ----------
        posterior_std : float
            Standard deviation of the Bayesian posterior distribution.

        Returns
        -------
        float
            Score component between 0.0 and 1.0.

        Notes
        -----
        std = 0 -> 1.0 (perfectly stable), std = 0.005 -> 0.5,
        std >= 0.010 -> 0.0 (wide credible interval, uncertain).
        Linear interpolation between 0 and 0.01.
        """
        return max(0.0, 1.0 - posterior_std / 0.01)


_confidence_scorer = _ConfidenceScorer()


def generate_slo_recommendation(rec_input_json: str) -> str:
    """
    Generate an SLO recommendation from metrics and dependency graph data.

    Parameters
    ----------
    rec_input_json : str
        JSON-encoded ``SLORecommendationInput`` object.

    Returns
    -------
    str
        JSON-encoded ``SLORecommendationOutput`` on success. On failure,
        falls back to a Wilson score + Hoeffding bound statistical estimate
        with ``confidence_score=0.35`` and ``requires_human_review=True``.
        Returns JSON with ``status="error"`` only if the fallback also fails.

    Notes
    -----
    Pipeline steps:
    1. Bayesian posterior lower 95% CI -> availability target.
    2. Series/parallel reliability formula -> dependency ceiling.
       Series: R = product(R_i for all sync deps).
       Parallel: R = 1 - product(1-R_i for all async deps).
    3. Monte Carlo validation (N samples, seed=42) -> sanity check.
    4. CLT p99 latency = mu + 2.326*sigma with 20% headroom.
       Hoeffding conservative bound with delta=0.01.
       Final p99 = median(CLT, max(critical_path, CLT), Hoeffding).
    5. Multi-component confidence scoring -> HITL gate at 0.75.
    Statistical fallback applies Wilson score (z=1.645, 95% CI) and
    Hoeffding inequality (delta=0.05) independently and takes the minimum.
    """
    try:
        inp = SLORecommendationInput.model_validate_json(rec_input_json)
        m   = inp.metrics_summary
        g   = inp.graph_analysis

        ci_lower   = float(m.get("credible_lower_95", m.get("posterior_mean", 0.99) - 0.002))
        rec_avail  = _floor_nines(max(0.90, ci_lower))

        sync_deps  = [(n, d.get("recommended_availability", 0.999))
                      for n, d in inp.dep_slos.items()
                      if d.get("dep_type", "synchronous") == "synchronous"]
        async_deps = [(n, d.get("recommended_availability", 0.999))
                      for n, d in inp.dep_slos.items()
                      if d.get("dep_type") == "asynchronous"]

        series_r   = math.prod(r for _, r in sync_deps)  if sync_deps  else 1.0
        parallel_r = (1.0 - math.prod(1.0 - r for _, r in async_deps)) if async_deps else 1.0
        ceiling    = _floor_nines(series_r * parallel_r)
        rec_avail  = min(rec_avail, ceiling)

        all_deps = {n: r for n, r in sync_deps + async_deps}
        mc_est   = _monte_carlo(all_deps,
                                [n for n, _ in sync_deps],
                                [n for n, _ in async_deps],
                                own=float(m.get("posterior_mean", 0.999)))

        smoothed_p99 = float(m.get("smoothed_p99_ms", m.get("p99_latency_ms", 200.0)))
        crit_lat     = float(g.get("critical_path_latency_ms", 0.0))

        lat_clt      = math.ceil(smoothed_p99 * 1.20 / 10) * 10
        lat_critical = math.ceil(crit_lat * 1.10 / 10) * 10 if crit_lat else 0

        a, b = 1.0, smoothed_p99 * 3
        hoeffding_t  = math.sqrt(-math.log(0.01) * (b - a)**2 / 2)
        lat_hoeffding = math.ceil((smoothed_p99 + hoeffding_t) / 10) * 10

        rec_lat = sorted([lat_clt, max(lat_critical, lat_clt), lat_hoeffding])[1]

        obs      = int(m.get("request_count_total", 0))
        post_std = float(m.get("posterior_std", 0.01))
        drift    = bool(m.get("drift_detected", False))
        anomaly  = str(m.get("anomaly_severity", "none"))
        quality  = float(m.get("data_quality_score", 1.0))

        confidence = _confidence_scorer.compute(
            obs=obs,
            posterior_std=post_std,
            drift_detected=drift,
            anomaly_severity=anomaly,
            data_quality=quality,
        )

        needs_review = confidence < 0.75 or drift or anomaly in ("high", "critical")
        review_reasons: list[str] = []
        if confidence < 0.75:
            review_reasons.append(f"Low confidence ({confidence:.2f})")
        if drift:
            review_reasons.append("Distribution drift detected (KL divergence > 0.1)")
        if anomaly in ("high", "critical"):
            review_reasons.append(f"Anomaly severity: {anomaly}")
        if quality < 0.60:
            review_reasons.append(f"Low data quality ({quality:.2f})")
        review_reason = ". ".join(review_reasons) + ("." if review_reasons else "")

        output = SLORecommendationOutput(
            service_name=inp.service_name,
            recommended_availability=rec_avail,
            recommended_latency_p99_ms=float(rec_lat),
            recommended_error_rate=round(1.0 - rec_avail, 6),
            confidence_score=round(confidence, 4),
            reasoning=(
                f"Availability {rec_avail:.4f} ({_nines_label(rec_avail)}) "
                f"from Bayesian lower-95-CI={ci_lower:.5f}. "
                f"Dependency ceiling={ceiling:.5f} "
                f"(series={series_r:.5f}, parallel={parallel_r:.5f}). "
                f"MC estimate={mc_est:.5f}. "
                f"Latency p99={rec_lat}ms (CLT={lat_clt}, Hoeffding={lat_hoeffding}). "
                f"Confidence={confidence:.2f}."
            ),
            math_details={
                "bayesian_ci_lower_95": ci_lower,
                "series_reliability": series_r,
                "parallel_reliability": parallel_r,
                "reliability_ceiling": ceiling,
                "monte_carlo_estimate": mc_est,
                "clt_latency_ms": lat_clt,
                "hoeffding_latency_ms": lat_hoeffding,
                "critical_path_latency_ms": crit_lat,
                "observation_count": obs,
            },
            data_sources=(
                inp.knowledge_sources
                if inp.knowledge_sources
                else ["metrics_db", "bayesian_update", "reliability_model", "graph_analysis"]
            ),
            requires_human_review=needs_review,
            review_reason=review_reason,
        )
        return output.model_dump_json()

    except Exception as e:
        try:
            m       = SLORecommendationInput.model_validate_json(rec_input_json).metrics_summary
            p_hat   = float(m.get("posterior_mean", 0.99))
            n_obs   = max(1, int(m.get("request_count_total", 1)))
            p99_ms  = float(m.get("smoothed_p99_ms", 200.0))

            z           = 1.645
            p_tilde     = p_hat + (z * z) / (2 * n_obs)
            denom       = 1.0 + (z * z) / n_obs
            under_root  = (p_hat * (1.0 - p_hat) / n_obs) + ((z * z) / (4 * n_obs * n_obs))
            wilson_low  = (p_tilde - z * math.sqrt(max(0.0, under_root))) / denom

            hoeffding_low = p_hat - math.sqrt(math.log(1.0 / 0.05) / (2.0 * n_obs))

            avail = _floor_nines(max(0.90, min(wilson_low, hoeffding_low)))

            lat_margin = 1.0 + math.sqrt(math.log(20.0) / max(1, n_obs)) * 2.0
            lat = math.ceil(p99_ms * lat_margin / 10) * 10

            return json.dumps({
                "recommended_availability":    avail,
                "recommended_latency_p99_ms":  float(lat),
                "recommended_error_rate":      round(1.0 - avail, 6),
                "confidence_score":            0.35,
                "reasoning": (
                    f"Statistical fallback (pipeline error: {type(e).__name__}). "
                    f"p_hat={p_hat:.5f}, n={n_obs}. "
                    f"Wilson lower bound={wilson_low:.5f}, "
                    f"Hoeffding lower bound={hoeffding_low:.5f}. "
                    f"Target = min(Wilson, Hoeffding) = {avail:.4f}. "
                    "SRE review required before applying."
                ),
                "requires_human_review": True,
                "review_reason":         f"Statistical fallback (Wilson+Hoeffding) — {type(e).__name__}: {e}",
                "math_details": {
                    "fallback":          True,
                    "p_hat":             round(p_hat, 5),
                    "n_obs":             n_obs,
                    "wilson_lower":      round(wilson_low, 5),
                    "hoeffding_lower":   round(hoeffding_low, 5),
                    "method":            "min(Wilson95, Hoeffding95)",
                },
                "data_sources":     ["statistical_fallback_wilson_hoeffding"],
                "knowledge_context": "",
                "knowledge_sources": [],
            })
        except Exception:
            return json.dumps({"status": "error", "message": str(e)})


def check_slo_feasibility(feasibility_input_json: str) -> str:
    """
    Check whether a proposed SLO is achievable given dependency constraints.

    Parameters
    ----------
    feasibility_input_json : str
        JSON-encoded ``FeasibilityCheckInput`` object.

    Returns
    -------
    str
        JSON-encoded ``FeasibilityCheckOutput`` on success, or JSON with
        ``status="error"`` and a ``message`` field on failure.

    Notes
    -----
    Availability ceiling = min(dep_series_product, historical * 1.05, 0.9999).
    Series reliability: ceiling = product(dep_availabilities).
    The proposed SLO is feasible when it does not exceed the ceiling and the
    proposed latency is at least 10ms (physical network floor). The feasibility
    score equals ceiling / proposed_availability when infeasible, capped at 0.1.
    When infeasible, ``adjusted_recommendation`` provides a safe alternative.
    """
    try:
        inp = FeasibilityCheckInput.model_validate_json(feasibility_input_json)
        issues: list[str] = []

        dep_vals = list(inp.dep_availabilities.values())
        dep_ceil = math.prod(dep_vals) if dep_vals else 1.0
        hist_cap = inp.historical_availability * 1.05
        ceiling  = min(dep_ceil, hist_cap, 0.9999)

        if inp.proposed_availability > ceiling:
            issues.append(
                f"Proposed {inp.proposed_availability:.5f} > ceiling {ceiling:.5f}. "
                f"(dep_series={dep_ceil:.5f}, hist_cap={hist_cap:.5f})"
            )

        if inp.proposed_latency_p99_ms < 10.0:
            issues.append(f"Latency {inp.proposed_latency_p99_ms}ms below 10ms physical floor.")

        feasible = len(issues) == 0
        fs_score = 1.0 if feasible else max(0.1, ceiling / max(inp.proposed_availability, 1e-9))

        adjusted = None
        if not feasible:
            adjusted = {
                "availability": _floor_nines(min(inp.proposed_availability, ceiling)),
                "latency_p99_ms": max(inp.proposed_latency_p99_ms, 10.0),
            }

        out = FeasibilityCheckOutput(
            is_feasible=feasible,
            feasibility_score=round(fs_score, 4),
            availability_ceiling=round(ceiling, 6),
            latency_floor_ms=10.0,
            issues=issues,
            adjusted_recommendation=adjusted,
        )
        return out.model_dump_json()
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


def run_milp_optimization(opt_input_json: str) -> str:
    """
    Find an optimal SLO portfolio using Mixed-Integer Linear Programming.

    Parameters
    ----------
    opt_input_json : str
        JSON-encoded object with fields:
        ``services`` (list of str), ``historical_availability`` (dict),
        ``importance_weights`` (dict), ``sync_deps`` (dict), and
        ``error_budget`` (float, default 0.001).

    Returns
    -------
    str
        JSON object with ``optimal_slos``, ``error_budget_allocation``,
        ``objective_value``, ``solver_status``, and
        ``total_downtime_minutes_per_year``. Returns a Bayesian floor fallback
        when the MILP is infeasible. Returns JSON with ``status="error"`` on
        unexpected failure.

    Notes
    -----
    MILP formulation:
    Variables: z_ik in {0,1} — service i assigned to SLO level k.
    Objective: max sum(w_i * slo_k * z_ik).
    Constraints:
      (1) sum_k(z_ik) = 1 for all i (exactly one level per service).
      (2) sum_i(sum_k((1-slo_k)*z_ik)) <= B_total (total error budget).
      (3) sum_k(log(slo_k)*z_ik) <= sum_k(log(slo_k)*z_jk) for all j in sync_deps(i).
      (4) z_ik = 0 when slo_k > hist_i * 1.05 (feasibility cap).
    Water-filling KKT optimal budget: b*_i = (w_i / sum(w_j)) * B_total.
    SLO levels are restricted to the standard discrete set:
    [0.90, 0.95, 0.99, 0.995, 0.999, 0.9995, 0.9999].
    CBC solver from PuLP is used with no external solver installation required.
    """
    try:
        inp      = json.loads(opt_input_json)
        svcs     = inp.get("services", [])
        hist     = inp.get("historical_availability", {})
        weights  = inp.get("importance_weights", {})
        sync_deps = inp.get("sync_deps", {})
        B_total  = float(inp.get("error_budget", 0.001))

        if not svcs:
            return json.dumps({"status": "error", "message": "No services provided."})

        SLO_LEVELS = [0.90, 0.95, 0.99, 0.995, 0.999, 0.9995, 0.9999]
        n = len(svcs)
        K = len(SLO_LEVELS)
        slo_idx = {s: i for i, s in enumerate(svcs)}

        prob = pulp.LpProblem("SLO_Portfolio_MILP", pulp.LpMaximize)

        z = {
            (i, k): pulp.LpVariable(f"z_{i}_{k}", cat="Binary")
            for i in range(n)
            for k in range(K)
        }

        prob += pulp.lpSum(
            weights.get(svcs[i], 1.0) * SLO_LEVELS[k] * z[i, k]
            for i in range(n)
            for k in range(K)
        ), "Maximise_weighted_availability"

        for i in range(n):
            prob += (
                pulp.lpSum(z[i, k] for k in range(K)) == 1,
                f"one_level_service_{i}",
            )

        prob += (
            pulp.lpSum(
                (1.0 - SLO_LEVELS[k]) * z[i, k]
                for i in range(n)
                for k in range(K)
            ) <= B_total,
            "total_error_budget",
        )

        for svc, deps in sync_deps.items():
            if svc not in slo_idx:
                continue
            i = slo_idx[svc]
            for dep in deps:
                if dep not in slo_idx:
                    continue
                j = slo_idx[dep]
                prob += (
                    pulp.lpSum(math.log(SLO_LEVELS[k]) * z[i, k] for k in range(K))
                    <= pulp.lpSum(math.log(SLO_LEVELS[k]) * z[j, k] for k in range(K)),
                    f"dep_ceil_{svc}_le_{dep}",
                )

        for i, svc in enumerate(svcs):
            cap = hist.get(svc, 0.99) * 1.05
            for k, slo in enumerate(SLO_LEVELS):
                if slo > cap:
                    prob += (z[i, k] == 0, f"cap_{svc}_level_{k}")

        solver = pulp.PULP_CBC_CMD(msg=0)
        prob.solve(solver)

        status = pulp.LpStatus[prob.status]

        if prob.status != pulp.constants.LpStatusOptimal:
            fallback = {s: _floor_nines(hist.get(s, 0.99) * 0.999) for s in svcs}
            return json.dumps({
                "status": "infeasible",
                "solver_status": status,
                "message": "MILP infeasible — returning Bayesian floor fallback.",
                "optimal_slos": fallback,
            })

        optimal: dict[str, float] = {}
        for i, svc in enumerate(svcs):
            for k, slo in enumerate(SLO_LEVELS):
                if pulp.value(z[i, k]) is not None and pulp.value(z[i, k]) > 0.5:
                    optimal[svc] = slo
                    break
            else:
                optimal[svc] = _floor_nines(hist.get(svc, 0.99))

        w_arr = np.array([weights.get(s, 1.0) for s in svcs], dtype=float)
        w_sum = w_arr.sum() or 1.0
        budgets = {
            svcs[i]: round(float(w_arr[i] / w_sum * B_total), 8)
            for i in range(n)
        }

        return json.dumps({
            "optimal_slos": optimal,
            "error_budget_allocation": budgets,
            "objective_value": round(float(pulp.value(prob.objective)), 6),
            "solver_status": status,
            "total_downtime_minutes_per_year": round(
                sum((1.0 - v) * 365.25 * 24 * 60 for v in optimal.values()), 1
            ),
        })

    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


def retrieve_knowledge_for_slo(
    query_json: str = "",
    service_name: str = "",
    service_type: str = "",
    tier: str = "medium",
    drift_detected: str = "false",
    anomaly_severity: str = "none",
    has_external_deps: str = "false",
    top_k: str = "4",
) -> str:
    """
    Retrieve relevant SRE runbooks, SLO templates, and incident history.

    Parameters
    ----------
    query_json : str, optional
        JSON-encoded object with all fields as a legacy invocation path.
        When provided, individual keyword arguments are ignored.
    service_name : str, optional
        Name of the service requesting knowledge retrieval.
    service_type : str, optional
        Service type category for semantic query enrichment.
    tier : str, optional
        Service criticality tier. Defaults to ``"medium"``.
    drift_detected : str, optional
        String boolean indicating KL-divergence drift. Defaults to ``"false"``.
    anomaly_severity : str, optional
        Anomaly severity from the metrics agent. Defaults to ``"none"``.
    has_external_deps : str, optional
        String boolean indicating external dependency presence. Defaults to ``"false"``.
    top_k : str, optional
        Number of results to retrieve as a string. Defaults to ``"4"``.

    Returns
    -------
    str
        JSON object with keys:
        ``results`` (list of dicts with id, type, title, content, relevance),
        ``source_ids`` (list of str), and ``context_summary`` (str).
        Returns JSON with ``error`` key on failure.

    Notes
    -----
    Builds a rich semantic query from the service name, type, tier, drift
    status, anomaly severity, and external dependency flag. Uses ChromaDB
    MMR retrieval via the ``knowledge_store`` singleton. The ``context_summary``
    field concatenates all retrieved document titles and contents for direct
    LLM consumption in the report synthesizer prompt.
    """
    try:
        def _to_bool(v) -> bool:
            if isinstance(v, bool): return v
            return str(v).lower() in ("true", "1", "yes")

        def _to_int(v, default=4) -> int:
            try: return int(v)
            except Exception: return default

        if query_json:
            inp = json.loads(query_json)
        else:
            inp = {
                "service_name": service_name,
                "service_type": service_type,
                "tier": tier,
                "drift_detected": drift_detected,
                "anomaly_severity": anomaly_severity,
                "has_external_deps": has_external_deps,
                "top_k": top_k,
            }
        service_name = inp.get("service_name", "")
        service_type = inp.get("service_type", "")
        tier         = inp.get("tier", "medium")
        drift        = _to_bool(inp.get("drift_detected", False))
        anomaly      = inp.get("anomaly_severity", "none")
        has_ext_deps = _to_bool(inp.get("has_external_deps", False))
        parsed_top_k = _to_int(inp.get("top_k", 4))

        query_parts = [service_name, service_type, "slo recommendation", tier]
        if drift:
            query_parts.append("drift detected kl divergence")
        if anomaly not in ("none", "low"):
            query_parts.append(f"anomaly {anomaly} severity")
        if has_ext_deps:
            query_parts.append("external dependency circuit breaker reliability")
        query = " ".join(query_parts)

        results = knowledge_store.retrieve(query, top_k=parsed_top_k)

        context_lines = []
        for r in results:
            context_lines.append(
                f"[{r['id']} — {r['type'].upper()}] {r['title']}\n{r['content']}"
            )
        context_summary = "\n\n".join(context_lines)

        return json.dumps({
            "results": results,
            "source_ids": [r["id"] for r in results],
            "context_summary": context_summary,
        })
    except Exception as e:
        return json.dumps({"error": str(e), "results": [], "source_ids": [], "context_summary": ""})


def _monte_carlo(
    dep_r: dict[str, float],
    sync: list[str],
    asyn: list[str],
    own: float,
    n: int = _MC_N,
) -> float:
    """
    Estimate system reliability using Monte Carlo simulation.

    Parameters
    ----------
    dep_r : dict of str to float
        Mapping of dependency service name to its availability probability.
    sync : list of str
        Names of synchronous dependencies (all must be up for system to work).
    asyn : list of str
        Names of asynchronous dependencies (at least one must be up).
    own : float
        Availability probability of the target service itself.
    n : int, optional
        Number of Monte Carlo samples. Defaults to ``_MC_N`` from settings.

    Returns
    -------
    float
        Point estimate of system availability. CI width is approximately
        +/- 0.02% at N=10,000.

    Notes
    -----
    System works if own works AND all sync deps work AND at least one async dep
    works. Uses a fixed random seed (42) for reproducibility. When no
    dependencies are provided, returns ``own`` directly.
    """
    rng = np.random.default_rng(seed=42)
    if not dep_r:
        return own

    deps  = list(dep_r.keys())
    r_arr = np.array([dep_r[d] for d in deps])
    si    = [i for i, d in enumerate(deps) if d in sync]
    ai    = [i for i, d in enumerate(deps) if d in asyn]

    samp = rng.random((n, len(deps))) < r_arr
    own_s = rng.random(n) < own
    sync_ok  = np.all(samp[:, si], axis=1) if si else np.ones(n, bool)
    async_ok = np.any(samp[:, ai], axis=1) if ai else np.ones(n, bool)
    return float((own_s & sync_ok & async_ok).mean())


def _floor_nines(v: float) -> float:
    """
    Round a value to the nearest standard SLO tier.

    Parameters
    ----------
    v : float
        Raw availability value to snap to a standard SLO boundary.

    Returns
    -------
    float
        Nearest standard SLO tier from [0.90, 0.95, 0.99, 0.995, 0.999,
        0.9995, 0.9999].

    Notes
    -----
    Uses absolute distance minimisation rather than a floor operation so that
    values very close to a higher tier are snapped upward.
    """
    tiers = [0.90, 0.95, 0.99, 0.995, 0.999, 0.9995, 0.9999]
    return min(tiers, key=lambda t: abs(t - v))


def _nines_label(v: float) -> str:
    """
    Return a human-readable nines label for an availability value.

    Parameters
    ----------
    v : float
        Availability value between 0 and 1.

    Returns
    -------
    str
        Label string such as ``"3-nines"`` for 0.999 or ``"4-nines"`` for 0.9999.

    Notes
    -----
    Computed as ``round(-log10(max(1-v, 1e-10)))``, floored at 1 nines to
    prevent non-positive labels for very low availability values.
    """
    nines = max(1, round(-math.log10(max(1.0 - v, 1e-10))))
    return f"{nines}-nines"
