"""
Prompts for the SLO recommendation agent workflow.

Notes
-----
ADK resolves ``{variable_name}`` placeholders from session state at runtime.
All state references must use ``{key}`` syntax, not ``state["key"]`` literal text.
The prompt strings are module-level constants imported by the agent module.
"""

slo_knowledge_tool_caller_prompt = """\
You are the SLO Knowledge Tool Caller.

Call retrieve_knowledge EXACTLY ONCE with these EXACT arguments — do not change them:
  query    = {rec_knowledge_query}
  top_k    = 4
  doc_type = "all"

Do NOT modify the query. Do NOT call any other tool. Do NOT add explanations or prose.
After retrieve_knowledge returns, output its result JSON verbatim as your final response.
"""

slo_knowledge_formatter_prompt = """\
The previous agent called retrieve_knowledge.
Find the function_response for retrieve_knowledge in the conversation history.

Extract and output ONLY a JSON object with these fields:
  source_ids:      list of source_id strings from the tool response
  context_summary: the context_summary string from the tool response
  total_returned:  the total_returned integer from the tool response
  kb_size:         the kb_size integer from the tool response

Output ONLY the JSON object. No other text.
If you cannot find the tool response, output: {"source_ids":[],"context_summary":"","total_returned":0,"kb_size":0}
"""

# Kept for backward compatibility
slo_knowledge_mcp_prompt = slo_knowledge_tool_caller_prompt

slo_gen_tool_caller_prompt = """\
You are the SLO generator tool caller. Call exactly these two tools in order and nothing else.

Service to analyse: {rec_service_name}
Metrics report:     {metrics_report_output}
Dependency report:  {dependency_report_output}

STEP 1 — Call retrieve_knowledge_for_slo with these individual keyword arguments:
  service_name     = {rec_service_name}
  service_type     = <infer from name: "checkout", "payment", "api_gateway", "auth", "database", "generic">
  tier             = <infer from dependency_report above, or "medium">
  drift_detected   = <"true" if drift_detected is true in metrics_report above, else "false">
  anomaly_severity = <anomaly_severity from metrics_report above, or "none">
  has_external_deps = <"true" if any dep_type is "external" in dependency_report above, else "false">
  top_k            = "4"

STEP 2 — After step 1 returns, call generate_slo_recommendation with:
  rec_input_json = a JSON string with these fields:
    service_name: {rec_service_name}
    metrics_summary: <dict from metrics_report above, or {{}}>
    graph_analysis: <dict from dependency_report above, or {{}}>
    dep_slos: {rec_dep_slos}
    knowledge_context: <context_summary from step 1 result>
    knowledge_sources: <source_ids list from step 1 result>

You MUST call generate_slo_recommendation. Do NOT skip it.

After generate_slo_recommendation returns, output its result JSON string verbatim as your final response.
Do NOT wrap it. Do NOT add prose. Output ONLY the raw JSON string returned by the tool.
"""

slo_gen_formatter_prompt = """\
The previous agent called generate_slo_recommendation.
In the conversation history you will find a function_response for generate_slo_recommendation.
That response has a "result" field whose value is a JSON string.

Output ONLY the value of that "result" field — the inner JSON string — verbatim.
Do NOT wrap it in any outer object. Do NOT add "result": around it.

Example: if the tool response was {"result": "{\"service_name\":\"foo\",...}"},
then output exactly: {"service_name":"foo",...}

If you cannot find it, output: {}
"""

slo_generator_prompt = slo_gen_tool_caller_prompt

slo_feasibility_tool_prompt = """\
You are the SLO Feasibility tool caller. Call check_slo_feasibility now.

Service: {rec_service_name}
Generation output: {rec_generation_output}
Metrics report: {metrics_report_output}

Call check_slo_feasibility with:
  feasibility_input_json = a JSON string with:
    service_name: {rec_service_name}
    proposed_availability: <recommended_availability from generation output above, or 0.99>
    proposed_latency_p99_ms: <recommended_latency_p99_ms from generation output above, or 200.0>
    historical_availability: <posterior_mean from metrics report above, or 0.99>
    dep_availabilities: {rec_dep_slos}

You MUST call check_slo_feasibility. Do not skip it.

After check_slo_feasibility returns, output its result JSON string verbatim as your final response.
Do NOT wrap it. Do NOT add prose. Output ONLY the raw JSON string returned by the tool.
"""

slo_feasibility_formatter_prompt = """\
The previous agent called check_slo_feasibility.
Find the function_response for check_slo_feasibility in the conversation history.
Extract the "result" field value (a JSON string) and output it verbatim.
Do NOT wrap it. Example: if response is {"result": "{\"is_feasible\":true,...}"},
output: {"is_feasible":true,...}
If you cannot find it, output: {}
"""

slo_feasibility_prompt = slo_feasibility_tool_prompt

slo_optimizer_tool_prompt = """\
You are the SLO Portfolio Optimizer tool caller. Call run_milp_optimization now.

Services list: {rec_services_list}
Historical availability: {rec_historical_availability}
Importance weights: {rec_importance_weights}

Call run_milp_optimization with:
  opt_input_json = a JSON string with:
    services: <list of service names from services list above>
    historical_availability: <dict from historical availability above, or {{}}>
    importance_weights: <dict from importance weights above, or {{}}>
    sync_deps: {{}}
    error_budget: 0.001

You MUST call run_milp_optimization. Do not skip it.

After run_milp_optimization returns, output its result JSON string verbatim as your final response.
Do NOT wrap it. Do NOT add prose. Output ONLY the raw JSON string returned by the tool.
"""

slo_optimizer_formatter_prompt = """\
The previous agent called run_milp_optimization.
Find the function_response for run_milp_optimization in the conversation history.
Extract the "result" field value (a JSON string) and output it verbatim.
Do NOT wrap it. Example: if response is {"result": "{\"optimal_slos\":{...}}"},
output: {"optimal_slos":{...}}
If you cannot find it, output: {}
"""

slo_optimizer_prompt = slo_optimizer_tool_prompt

slo_report_prompt = """\
You are the SLO Recommendation Report synthesizer.

Service: {rec_service_name}

Dependency analysis (PageRank, blast-radius, SCC cycles, SLO ceiling notes):
{dependency_report_output}

Metrics analysis (Bayesian posterior, KL drift, anomaly, error budget):
{metrics_report_output}

Retrieved knowledge (runbooks, SLO templates, incident history):
{rec_knowledge_output}

SLO generation result (Bayesian CI + series/parallel reliability + Monte Carlo + CLT):
{rec_generation_output}

Feasibility check result (LP feasibility, dep ceiling, adjusted recommendation):
{rec_feasibility_output}

MILP optimizer result (Pareto-optimal SLOs, error budget allocation — empty if single service):
{rec_optimizer_output}

Fill ALL fields of SLORecommendationReport:

- service_name: {rec_service_name}
- recommended_availability: use recommended_availability from generation result.
    IF feasibility result says is_feasible=false AND adjusted_recommendation exists,
    use adjusted_recommendation.availability instead.
- recommended_latency_p99_ms: from generation result (same override rule as above).
- recommended_error_rate: 1.0 - recommended_availability, rounded to 6 decimal places.
- confidence_score: from generation result.
- is_feasible: from feasibility result (default true if missing).
- feasibility_score: from feasibility result (default 1.0 if missing).
- availability_ceiling: from feasibility result (default 0.9999 if missing).
- requires_human_review: from generation result.
- review_reason: from generation result (empty string if none).
- pareto_optimal_slos: from optimizer result optimal_slos.
    If optimizer result is empty or missing, use {{"rec_service_name": recommended_availability}}.
- error_budget_allocation: from optimizer result error_budget_allocation (empty dict if missing).
- math_details: from generation result math_details (empty dict if missing).
- reasoning: from generation result reasoning.
- summary: 2-3 sentence engineer-facing summary combining availability target, confidence,
    whether human review is needed, and the most important signal from dep/metrics analysis.
- sources: source_ids list from knowledge result (empty list if missing).
- knowledge_context: context_summary from knowledge result (empty string if missing).

Output raw JSON only.
Do NOT use markdown fences.
Do NOT wrap the JSON in ```json ... ```.
Do NOT add prose before or after the JSON.
Respond with valid JSON matching SLORecommendationReport. No other text.
"""

slo_orchestrator_prompt = """\
You are the SLO Recommendation Orchestrator. Drive the full SLO generation pipeline:

Workflow steps:
1. GENERATE    — Bayesian + reliability + Monte Carlo + CLT recommendation
2. AWAIT_INPUT — Waiting for human tier assignment (pipeline paused for new services)
3. FEASIBILITY — Feasibility check + dependency ceiling validation
4. OPTIMIZE    — MILP portfolio optimization (only if multiple services)
5. REPORT      — Final SLO recommendation report

Current step: {rec_workflow_step}
Human answers (if any): {rec_user_answers}
"""
