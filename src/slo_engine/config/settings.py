"""
Application configuration via dynaconf.

Notes
-----
Layering order (later overrides earlier):
  1. settings.toml [default] section.
  2. settings.toml [<ENV_FOR_DYNACONF>] section (default: development).
  3. .secrets.toml (never committed to source control).
  4. Environment variables prefixed SLO_, e.g.:
     SLO_POSTGRES__DSN="postgresql+asyncpg://..."
     SLO_LLM__MODEL="ollama/llama3.1"
     SLO_COMPUTE_LOGIC__MONTE_CARLO_SAMPLES=50000

Switch environment with: export ENV_FOR_DYNACONF=production

Nested key access uses double underscores as separators:
  SLO_API__JWT_SECRET=... -> settings.api.jwt_secret
"""
from __future__ import annotations

from pathlib import Path

from dynaconf import Dynaconf, Validator

_ROOT = Path(__file__).resolve().parents[3]

settings = Dynaconf(
    envvar_prefix="SLO",
    env_switcher="ENV_FOR_DYNACONF",

    settings_files=[
        str(_ROOT / "settings.toml"),
        str(_ROOT / ".secrets.toml"),
    ],

    environments=True,
    load_dotenv=False,
    merge_enabled=True,
    lowercase_read=True,

    validators=[
        Validator("api.jwt_secret", len_min=32,
                  messages={"len_min": "api.jwt_secret must be >= 32 chars — set SLO_API__JWT_SECRET"}),

        Validator("compute_logic.pagerank_damping", gt=0.0, lt=1.0),

        Validator("compute_logic.kalman_process_noise",     gt=0.0),
        Validator("compute_logic.kalman_measurement_noise", gt=0.0),

        Validator("compute_logic.monte_carlo_samples", gte=100),

        Validator("compute_logic.bayesian_prior_alpha", gt=0.0),
        Validator("compute_logic.bayesian_prior_beta",  gt=0.0),

        Validator("hitl.confidence_threshold", gt=0.0, lt=1.0),

        Validator("api.port",              gte=1024, lte=65535),
        Validator("mcp.metrics_port",      gte=1024, lte=65535),
        Validator("mcp.dependency_port",   gte=1024, lte=65535),
        Validator("mcp.knowledge_port",    gte=1024, lte=65535),
    ],
)


def get_settings() -> Dynaconf:
    """
    Return the validated settings instance.

    Returns
    -------
    Dynaconf
        The module-level ``settings`` singleton, already validated on import.

    Notes
    -----
    This function exists as a convenience alias to avoid deep attribute chains
    in calling code. The Dynaconf instance is a module-level singleton and is
    validated at import time — calling this function does not re-validate.
    """
    return settings
