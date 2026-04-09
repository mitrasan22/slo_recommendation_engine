"""
SLO pipeline full orchestration app exposed to ADK.

Notes
-----
Wraps the root ``SLORouterAgent`` (dependency -> metrics -> recommendation -> gate).
ADK discovers this subdirectory and serves it as ``app_name="slo_pipeline"``.

Usage::

    POST /apps/slo_pipeline/users/{user_id}/sessions  {}
    POST /run  {
        "app_name": "slo_pipeline",
        "user_id": ...,
        "session_id": ...,
        "new_message": {
            "role": "user",
            "parts": [{"text": "Recommend SLO for checkout-service"}]
        }
    }
"""
from slo_engine.agents.agent import build_root_agent

root_agent = build_root_agent()
