"""
LLM registry for the SLO Recommendation Engine.

Notes
-----
Registers LLM backends via Google ADK (native) and LiteLLM (alternatives).

Default backend: Google Gemini 3 Flash Preview (native ADK integration).
GEMINI_API_KEY must be set in .env and is picked up automatically by ADK.

Alternative backends (selected by model string prefix in LLM_MODEL env var):

  ollama/...       -> OllamaLiteLlm  (local, requires Ollama daemon)
                     e.g. "ollama/mistral", "ollama/llama3.1"

  huggingface/...  -> HuggingFaceLiteLlm  (HuggingFace Serverless Inference API)
                     e.g. "huggingface/mistralai/Mistral-7B-Instruct-v0.3"
                     Requires: HUGGINGFACE_API_KEY env var (free HF account)

  openai/...       -> LiteLlm default  (OpenAI or any OpenAI-compatible endpoint)
                     e.g. "openai/gpt-4o-mini"
                     Requires: OPENAI_API_KEY env var

Gemini models always use native ADK integration. If a Gemini model string is
accidentally provided with a ``gemini/`` prefix, it is normalised elsewhere in
the codebase to the bare ADK form (for example
``"gemini/gemini-3.0-flash-preview"`` -> ``"gemini-3.0-flash-preview"``).

Switching backends: set LLM_MODEL env var or change llm.model in settings.toml.
"""
from __future__ import annotations

import os

import litellm
from google.adk.models.lite_llm import LiteLlm
from google.adk.models.registry import LLMRegistry
from loguru import logger
from typing_extensions import override

logger = logger.bind(name=__name__)
litellm.modify_params = True
litellm.num_retries = 5
litellm.retry_after = 65


class OllamaLiteLlm(LiteLlm):
    """
    LiteLlm subclass for locally-running Ollama models.

    Notes
    -----
    Requires the Ollama daemon running at the URL specified in the
    ``OLLAMA_API_BASE`` environment variable (default: http://localhost:11434).
    No API key is required. The model must be pulled first with
    ``ollama pull <model-name>`` before use.
    """

    @classmethod
    @override
    def supported_models(cls) -> list[str]:
        """
        Return the list of model string patterns handled by this backend.

        Returns
        -------
        list of str
            Regex patterns matching ollama model strings including
            ``"ollama/.*"``, ``"ollama_chat/.*"``, and bare model names.

        Notes
        -----
        Patterns are used by the LLMRegistry to route model strings to
        the correct LiteLlm subclass.
        """
        return [
            r"ollama/.*",
            r"ollama_chat/.*",
            r"mistral",
            r"llama.*",
            r"mixtral.*",
        ]


class HuggingFaceLiteLlm(LiteLlm):
    """
    LiteLlm subclass for the HuggingFace Serverless Inference API.

    Notes
    -----
    Requires a HuggingFace account and API token set in the
    ``HUGGINGFACE_API_KEY`` environment variable. LiteLLM picks up the
    key automatically. The free tier provides approximately 1000
    requests per day, sufficient for demonstration purposes.

    Recommended free models include:
    ``huggingface/mistralai/Mistral-7B-Instruct-v0.3``,
    ``huggingface/HuggingFaceH4/zephyr-7b-beta``,
    ``huggingface/microsoft/Phi-3.5-mini-instruct``,
    ``huggingface/google/gemma-2-2b-it``.
    """

    @classmethod
    @override
    def supported_models(cls) -> list[str]:
        """
        Return the list of model string patterns handled by this backend.

        Returns
        -------
        list of str
            Regex patterns matching HuggingFace model strings of the
            form ``"huggingface/<org>/<model-name>"``.

        Notes
        -----
        Patterns are used by the LLMRegistry to route model strings to
        the correct LiteLlm subclass.
        """
        return [
            r"huggingface/.*",
        ]


LLMRegistry.register(OllamaLiteLlm)
LLMRegistry.register(HuggingFaceLiteLlm)

_model = os.getenv("LLM_MODEL", "gemini-3-flash-preview")
if _model.startswith("huggingface/"):
    _key_set = bool(os.getenv("HUGGINGFACE_API_KEY"))
    logger.info(
        "LLM backend: HuggingFace Inference API | model={} | api_key_set={}",
        _model,
        _key_set,
    )
    if not _key_set:
        logger.warning(
            "HUGGINGFACE_API_KEY not set — HuggingFace calls will fail. "
            "Get a free token at https://huggingface.co/settings/tokens"
        )
elif _model.startswith("ollama/"):
    logger.info("LLM backend: Ollama | model={}", _model)
else:
    logger.info("LLM backend: Gemini (native ADK) | model={}", _model)
