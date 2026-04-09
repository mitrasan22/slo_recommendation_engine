"""
Base agent abstraction for the SLO Recommendation Engine.

Notes
-----
Provides a typed, subclassable wrapper around Google ADK agent types. All
concrete agents inherit from BaseAgent and declare class-level attributes to
configure their name, model, tools, sub-agents, callbacks, and orchestration
type. The build() method constructs the appropriate ADK agent instance.
"""
from __future__ import annotations

import inspect
from collections.abc import AsyncGenerator, Sequence
from typing import Any, Literal, TypeAlias, TypeVar, cast

from dotenv import load_dotenv
from google.adk.agents import Agent, LoopAgent, ParallelAgent, SequentialAgent
from google.adk.agents import BaseAgent as ADKBaseAgent
from google.adk.agents.base_agent import AfterAgentCallback, BeforeAgentCallback
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import (
    AfterModelCallback,
    AfterToolCallback,
    BeforeModelCallback,
    BeforeToolCallback,
    InstructionProvider,
    ToolUnion,
)
from google.adk.code_executors.base_code_executor import BaseCodeExecutor
from google.adk.events.event import Event
from google.adk.models.base_llm import BaseLlm
from google.adk.planners.base_planner import BasePlanner
from google.genai import types as genai_types
from loguru import logger
from pydantic import BaseModel

from slo_engine.agents.llm_manager import LLMRegistry
from slo_engine.config import config
from slo_engine.core.agent_registry import AgentRegistry

OnModelErrorCallback: TypeAlias = Any
OnToolErrorCallback: TypeAlias = Any

logger = logger.bind(name=__name__)
load_dotenv()

_tool_logger = logger.bind(name="slo_engine.agents.tool_trace")


def _log_tool_call(tool, args, tool_context):
    """
    Default before_tool_callback that logs every tool invocation.

    Parameters
    ----------
    tool : Any
        The ADK tool object being invoked.
    args : dict
        Arguments passed to the tool.
    tool_context : Any
        ADK tool context object.

    Returns
    -------
    None
        Returning ``None`` instructs ADK to proceed normally without
        modifying the arguments.

    Notes
    -----
    Logs at INFO level with the tool name and full argument dictionary.
    """
    _tool_logger.info(
        "TOOL> {} | args={}",
        getattr(tool, "name", str(tool)),
        args,
    )
    return None


def _log_tool_result(tool, args, tool_context, tool_response):
    """
    Default after_tool_callback that logs every tool result.

    Parameters
    ----------
    tool : Any
        The ADK tool object that was invoked.
    args : dict
        Arguments that were passed to the tool.
    tool_context : Any
        ADK tool context object.
    tool_response : Any
        The response returned by the tool.

    Returns
    -------
    None
        Returning ``None`` instructs ADK to use the original response unmodified.

    Notes
    -----
    Truncates the result string representation to 3000 characters to avoid
    flooding the log file with large JSON payloads.
    """
    _tool_logger.info(
        "TOOL< {} | result={}",
        getattr(tool, "name", str(tool)),
        str(tool_response)[:3000],
    )
    return None


def _log_llm_request(callback_context, llm_request):
    """
    Default before_model_callback that logs the system instruction and input messages.

    Parameters
    ----------
    callback_context : Any
        ADK callback context providing the agent name.
    llm_request : Any
        ADK LLM request object containing config and contents.

    Returns
    -------
    None
        Returning ``None`` instructs ADK to proceed normally without
        modifying the request.

    Notes
    -----
    Logs the system instruction and each input message at INFO level.
    Exceptions during introspection are silently ignored to ensure the
    callback never breaks the agent pipeline.
    """
    agent_name = getattr(callback_context, "agent_name", "?")
    try:
        sys_parts = getattr(llm_request.config, "system_instruction", None)
        if sys_parts:
            text = getattr(sys_parts, "text", None) or " ".join(
                getattr(p, "text", "") for p in getattr(sys_parts, "parts", []) if hasattr(p, "text")
            )
            _tool_logger.info("LLM> [{}] SYSTEM:\n{}", agent_name, text)
    except Exception:
        pass
    try:
        for msg in (llm_request.contents or []):
            role = getattr(msg, "role", "?")
            parts = getattr(msg, "parts", []) or []
            text  = " ".join(getattr(p, "text", "") for p in parts if hasattr(p, "text") and getattr(p, "text", ""))
            if text:
                _tool_logger.info("LLM> [{}] {}: {}", agent_name, role.upper(), text)
    except Exception:
        pass
    return None


def _log_llm_response(callback_context, llm_response):
    """
    Default after_model_callback that logs the full model response text.

    Parameters
    ----------
    callback_context : Any
        ADK callback context providing the agent name.
    llm_response : Any
        ADK LLM response object containing the model output content.

    Returns
    -------
    None
        Returning ``None`` instructs ADK to use the original response unmodified.

    Notes
    -----
    Extracts text from all parts of the response content. Exceptions during
    introspection are silently ignored.
    """
    agent_name = getattr(callback_context, "agent_name", "?")
    try:
        content = getattr(llm_response, "content", None)
        parts   = getattr(content, "parts", []) or [] if content else []
        text    = " ".join(getattr(p, "text", "") for p in parts if hasattr(p, "text") and getattr(p, "text", ""))
        if text:
            _tool_logger.info("LLM< [{}] MODEL:\n{}", agent_name, text)
    except Exception:
        pass
    return None

OrchestrationType = Literal["single", "sequential", "parallel", "loop"]
ADKAgentType = Agent | SequentialAgent | ParallelAgent | LoopAgent
RunAsyncOverride = Any

_WARNED_UNSUPPORTED_AGENT_KWARGS: set[str] = set()
_WARNED_RUN_OVERRIDE_CONFLICT: set[type[BaseAgent]] = set()
_WARNED_JSON_SCHEMA_OVERRIDES: set[str] = set()


_SelfBaseAgent = TypeVar("_SelfBaseAgent", bound="BaseAgent")
_ADKAgentT = TypeVar("_ADKAgentT", bound=ADKBaseAgent)


def _normalize_model_string(model_name: str) -> str:
    """
    Normalise model identifiers before backend resolution.

    Notes
    -----
    Gemini models should use native Google ADK model names. Accept the
    accidental LiteLLM-style prefix for compatibility and strip it.
    """
    if model_name.startswith("gemini/"):
        return model_name.split("/", 1)[1]
    return model_name


class BaseAgent:
    """
    Typed, subclassable wrapper around Google ADK agent types.

    Attributes
    ----------
    name : str
        Unique agent identifier used in ADK routing and logs.
    description : str
        Human-readable description for ADK agent discovery.
    orchestration : OrchestrationType
        Agent composition mode: ``"single"``, ``"sequential"``, ``"parallel"``,
        or ``"loop"``.
    model : str or BaseLlm or None
        LLM model string or instance. Resolved via LLMRegistry if a string.
    instruction : str or InstructionProvider
        System prompt or callable that returns the system prompt.
    tools : Sequence[ToolUnion]
        List of ADK-compatible tool functions or FunctionTool instances.
    sub_agents : Sequence[ADKBaseAgent]
        List of child ADK agents for composite orchestration types.
    generate_content_config : GenerateContentConfig or None
        ADK generation config override.
    input_schema : type[BaseModel] or None
        Pydantic model for structured input validation.
    output_schema : type[BaseModel] or None
        Pydantic model for structured output parsing.
    response_json_schema : dict or None
        Raw JSON schema override for response format.
    output_key : str or None
        Session state key where the agent stores its output.
    disallow_transfer_to_parent : bool
        If True, the agent cannot transfer control back to its parent.
    disallow_transfer_to_peers : bool
        If True, the agent cannot transfer control to sibling agents.
    include_contents : Literal
        Whether to include conversation history in LLM requests.
    max_iterations : int or None
        Maximum loop iterations for ``"loop"`` orchestration.
    before_agent_callback : BeforeAgentCallback or None
        Callback invoked before the agent processes each turn.
    after_agent_callback : AfterAgentCallback or None
        Callback invoked after the agent processes each turn.
    before_model_callback : BeforeModelCallback or None
        Callback invoked before each LLM call. Defaults to request logger.
    after_model_callback : AfterModelCallback or None
        Callback invoked after each LLM call. Defaults to response logger.
    before_tool_callback : BeforeToolCallback or None
        Callback invoked before each tool call. Defaults to tool call logger.
    after_tool_callback : AfterToolCallback or None
        Callback invoked after each tool call. Defaults to tool result logger.
    planner : BasePlanner or None
        Optional ADK planner for multi-step reasoning.
    code_executor : BaseCodeExecutor or None
        Optional ADK code executor for code generation tasks.
    auto_register : bool
        If True, registers the built agent in AgentRegistry automatically.

    Notes
    -----
    Subclasses declare class-level attributes to configure the agent, then call
    ``build()`` to produce the corresponding ADK agent instance. Subclasses that
    implement custom logic override ``run_async_impl``.
    """

    name: str = ""
    description: str = ""
    orchestration: OrchestrationType = "single"
    model: str | BaseLlm | None = None
    instruction: str | InstructionProvider = ""
    tools: Sequence[ToolUnion] = ()
    sub_agents: Sequence[ADKBaseAgent] = ()
    generate_content_config: genai_types.GenerateContentConfig | None = None
    input_schema: type[BaseModel] | None = None
    output_schema: type[BaseModel] | None = None
    response_json_schema: dict[str, Any] | None = None
    output_key: str | None = None
    disallow_transfer_to_parent: bool = False
    disallow_transfer_to_peers: bool = False
    include_contents: Literal["default", "none"] = "default"
    max_iterations: int | None = None
    before_agent_callback: BeforeAgentCallback | None = None
    after_agent_callback: AfterAgentCallback | None = None
    before_model_callback: BeforeModelCallback | None = _log_llm_request
    after_model_callback: AfterModelCallback | None = _log_llm_response
    on_model_error_callback: OnModelErrorCallback | None = None
    before_tool_callback: BeforeToolCallback | None = _log_tool_call
    after_tool_callback: AfterToolCallback | None = _log_tool_result
    on_tool_error_callback: OnToolErrorCallback | None = None
    planner: BasePlanner | None = None
    code_executor: BaseCodeExecutor | None = None
    auto_register: bool = True

    def __init__(self, **overrides: Any):
        """
        Initialise the agent, applying any attribute overrides.

        Parameters
        ----------
        **overrides : Any
            Keyword arguments that override class-level attribute defaults.

        Notes
        -----
        Raises ``TypeError`` if an override key is not a declared attribute.
        Raises ``ValueError`` if ``name`` is empty after applying overrides.
        """
        for key, value in overrides.items():
            if not hasattr(self.__class__, key) and not hasattr(self, key):
                raise TypeError(f"BaseAgent has no attribute '{key}'")
            setattr(self, key, value)
        if not self.name:
            raise ValueError("BaseAgent requires a non-empty 'name'.")

    @classmethod
    def create(cls: type[_SelfBaseAgent], **overrides: Any) -> _SelfBaseAgent:
        """
        Create an instance of this agent class with optional overrides.

        Parameters
        ----------
        **overrides : Any
            Keyword arguments forwarded to ``__init__``.

        Returns
        -------
        _SelfBaseAgent
            New instance of the subclass.

        Notes
        -----
        Equivalent to calling the constructor directly. Provided as a
        class method for consistency with the factory pattern used by
        ``single``, ``sequential``, ``parallel``, and ``loop``.
        """
        return cls(**overrides)

    @classmethod
    def single(cls: type[_SelfBaseAgent], **overrides: Any) -> _SelfBaseAgent:
        """
        Create an instance configured for single-LLM orchestration.

        Parameters
        ----------
        **overrides : Any
            Keyword arguments forwarded to ``__init__``.

        Returns
        -------
        _SelfBaseAgent
            New instance with ``orchestration="single"``.

        Notes
        -----
        Overrides any ``orchestration`` value declared on the class.
        """
        return cls.create(orchestration="single", **overrides)

    @classmethod
    def sequential(cls: type[_SelfBaseAgent], **overrides: Any) -> _SelfBaseAgent:
        """
        Create an instance configured for sequential orchestration.

        Parameters
        ----------
        **overrides : Any
            Keyword arguments forwarded to ``__init__``.

        Returns
        -------
        _SelfBaseAgent
            New instance with ``orchestration="sequential"``.

        Notes
        -----
        Builds a SequentialAgent that runs sub-agents in order.
        """
        return cls.create(orchestration="sequential", **overrides)

    @classmethod
    def parallel(cls: type[_SelfBaseAgent], **overrides: Any) -> _SelfBaseAgent:
        """
        Create an instance configured for parallel orchestration.

        Parameters
        ----------
        **overrides : Any
            Keyword arguments forwarded to ``__init__``.

        Returns
        -------
        _SelfBaseAgent
            New instance with ``orchestration="parallel"``.

        Notes
        -----
        Builds a ParallelAgent that runs sub-agents concurrently.
        """
        return cls.create(orchestration="parallel", **overrides)

    @classmethod
    def loop(cls: type[_SelfBaseAgent], **overrides: Any) -> _SelfBaseAgent:
        """
        Create an instance configured for loop orchestration.

        Parameters
        ----------
        **overrides : Any
            Keyword arguments forwarded to ``__init__``.

        Returns
        -------
        _SelfBaseAgent
            New instance with ``orchestration="loop"``.

        Notes
        -----
        Builds a LoopAgent that iterates sub-agents up to ``max_iterations``.
        """
        return cls.create(orchestration="loop", **overrides)

    def resolve_model(self) -> BaseLlm | str | None:
        """
        Resolve the configured model string or instance to an ADK BaseLlm.

        Returns
        -------
        BaseLlm or None
            Resolved LLM instance ready for use in ADK.

        Notes
        -----
        If ``self.model`` is already a ``BaseLlm``, it is returned unchanged.
        If it is a string, ``LLMRegistry.resolve`` is used to instantiate it.
        Falls back to ``config.model.model_string`` if neither is set.
        Raises ``ValueError`` if no model can be resolved.
        """
        if isinstance(self.model, BaseLlm):
            return self.model
        elif isinstance(self.model, str) and self.model:
            model_name = _normalize_model_string(self.model)
            if model_name.startswith(("ollama/", "ollama_chat/", "huggingface/", "openai/")):
                return LLMRegistry.resolve(model_name)(model=model_name)
            return model_name
        elif (m := getattr(config, "model", None)) and isinstance(m, dict):
            model_name = _normalize_model_string(m["model_string"])
            if model_name.startswith(("ollama/", "ollama_chat/", "huggingface/", "openai/")):
                return LLMRegistry.resolve(model_name)(model=model_name)
            return model_name
        else:
            raise ValueError("Model string not set in agent or config.")

    def get_instruction(self) -> str | InstructionProvider:
        """
        Return the agent's instruction string or provider.

        Returns
        -------
        str or InstructionProvider
            The system prompt or callable instruction provider.

        Notes
        -----
        Subclasses may override this method to compute instructions dynamically.
        """
        return self.instruction

    def get_tools(self) -> list[ToolUnion]:
        """
        Return the list of tools available to this agent.

        Returns
        -------
        list of ToolUnion
            Materialised list of tool functions or FunctionTool instances.

        Notes
        -----
        Subclasses may override this method to compose tools dynamically.
        """
        return list(self.tools)

    def get_sub_agents(self) -> Sequence[ADKBaseAgent]:
        """
        Return the list of sub-agents for composite orchestration.

        Returns
        -------
        Sequence of ADKBaseAgent
            List of child ADK agent instances.

        Notes
        -----
        Subclasses may override this method to compose sub-agents dynamically.
        """
        return list(self.sub_agents)

    async def run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        """
        Override point for custom agent logic that yields ADK events.

        Parameters
        ----------
        ctx : InvocationContext
            ADK invocation context with session state and user message.

        Returns
        -------
        AsyncGenerator[Event, None]
            Async generator of ADK events.

        Notes
        -----
        Raises ``NotImplementedError`` by default. Subclasses that need custom
        orchestration logic must override this method.
        """
        raise NotImplementedError("run_async_impl is not implemented for this agent.")
        yield  # pragma: no cover

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        """
        Internal run implementation that delegates to ``run_async_impl``.

        Parameters
        ----------
        ctx : InvocationContext
            ADK invocation context.

        Returns
        -------
        AsyncGenerator[Event, None]
            Async generator of ADK events from ``run_async_impl``.

        Notes
        -----
        Exists as the ADK-recognised override point. Delegates to the public
        ``run_async_impl`` which subclasses override directly.
        """
        async for event in self.run_async_impl(ctx):
            yield event

    def build(self) -> ADKAgentType:
        """
        Build and return the configured ADK agent instance.

        Returns
        -------
        ADKAgentType
            The constructed ADK Agent, SequentialAgent, ParallelAgent, or LoopAgent.

        Notes
        -----
        Calls ``_build_agent()`` and then registers the result in ``AgentRegistry``
        if ``auto_register`` is True.
        """
        agent = self._build_agent()
        if self.auto_register:
            AgentRegistry.register(self.name, agent)
        return agent

    def _build_agent(self) -> ADKAgentType:
        """
        Dispatch to the appropriate build method based on orchestration type.

        Returns
        -------
        ADKAgentType
            The constructed ADK agent.

        Notes
        -----
        Raises ``ValueError`` for unrecognised orchestration type values.
        """
        sub_agents = self.get_sub_agents()
        run_override = self._resolve_run_override()
        if self.orchestration == "single":
            return self._build_single(sub_agents, run_override)
        if self.orchestration == "sequential":
            return self._build_composite(SequentialAgent, sub_agents, run_override)
        if self.orchestration == "parallel":
            return self._build_composite(ParallelAgent, sub_agents, run_override)
        if self.orchestration == "loop":
            return self._build_loop(sub_agents, run_override)
        raise ValueError(f"Invalid orchestration type: {self.orchestration!r}")

    def _build_single(self, sub_agents: Sequence[ADKBaseAgent], run_override: RunAsyncOverride | None) -> Agent:
        """
        Build a single-LLM ADK Agent with all configured parameters.

        Parameters
        ----------
        sub_agents : Sequence of ADKBaseAgent
            Child agent instances to pass to the ADK Agent constructor.
        run_override : callable or None
            Optional run_async_impl override to inject into the agent class.

        Returns
        -------
        Agent
            Constructed ADK Agent instance.

        Notes
        -----
        Filters kwargs against the set of fields supported by the installed
        version of google-adk to allow forward/backward compatibility.
        Unsupported kwargs are logged once at WARNING level.
        """
        kwargs: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "model": self.resolve_model(),
            "instruction": self.get_instruction(),
            "tools": self.get_tools(),
            "sub_agents": sub_agents,
            "generate_content_config": self._resolve_generate_content_config(),
            "disallow_transfer_to_parent": self.disallow_transfer_to_parent,
            "disallow_transfer_to_peers": self.disallow_transfer_to_peers,
            "include_contents": self.include_contents,
            "before_agent_callback": self._resolve_callback("before_agent_callback"),
            "after_agent_callback": self._resolve_callback("after_agent_callback"),
            "before_model_callback": self._resolve_callback("before_model_callback"),
            "after_model_callback": self._resolve_callback("after_model_callback"),
            "on_model_error_callback": self._resolve_callback("on_model_error_callback"),
            "before_tool_callback": self._resolve_callback("before_tool_callback"),
            "after_tool_callback": self._resolve_callback("after_tool_callback"),
            "on_tool_error_callback": self._resolve_callback("on_tool_error_callback"),
        }
        if self.input_schema is not None:
            kwargs["input_schema"] = self.input_schema
        if self.output_key is not None:
            kwargs["output_key"] = self.output_key
        if self.planner is not None:
            kwargs["planner"] = self.planner
        if self.code_executor is not None:
            kwargs["code_executor"] = self.code_executor
        supported_fields = self._supported_agent_fields()
        if not supported_fields:
            return self._instantiate_with_optional_run_override(Agent, kwargs, run_override)
        filtered: dict[str, Any] = {}
        for key, value in kwargs.items():
            if key in supported_fields:
                filtered[key] = value
            elif value is not None and not self._is_default_optional_error_callback(key, value):
                self._warn_unsupported_kwarg_once(key)
        return self._instantiate_with_optional_run_override(Agent, filtered, run_override)

    def _resolve_generate_content_config(self) -> genai_types.GenerateContentConfig | None:
        """
        Build the GenerateContentConfig merging response_json_schema if set.

        Returns
        -------
        GenerateContentConfig or None
            Resolved generation configuration for the LLM call.

        Notes
        -----
        When ``response_json_schema`` is set, it is merged into the config and
        ``response_schema`` is cleared to avoid conflict.
        """
        resolved_response_json_schema = self._resolve_response_json_schema()
        if resolved_response_json_schema is None:
            return self.generate_content_config
        if self.generate_content_config is None:
            return genai_types.GenerateContentConfig(response_json_schema=resolved_response_json_schema)
        return self.generate_content_config.model_copy(
            update={"response_json_schema": resolved_response_json_schema, "response_schema": None}
        )

    def _resolve_response_json_schema(self) -> dict[str, Any] | None:
        """
        Resolve the response JSON schema for Gemini-compatible structured output.

        Notes
        -----
        Prefer an explicitly provided ``response_json_schema``. Otherwise,
        derive JSON Schema from the Pydantic ``output_schema`` so agents can
        use ``response_json_schema`` without relying on deprecated
        ``response_schema`` plumbing.
        """
        if self.response_json_schema is not None:
            return self.response_json_schema
        if self.output_schema is None:
            return None
        try:
            return cast(dict[str, Any], self.output_schema.model_json_schema())
        except Exception:
            return None

    def _build_composite(self, agent_cls, sub_agents, run_override):
        """
        Build a composite ADK agent (SequentialAgent or ParallelAgent).

        Parameters
        ----------
        agent_cls : type
            The ADK composite agent class to instantiate.
        sub_agents : list
            Child agents to pass to the composite constructor.
        run_override : callable or None
            Optional run_async_impl override.

        Returns
        -------
        SequentialAgent or ParallelAgent
            Constructed composite ADK agent.

        Notes
        -----
        Raises ``ValueError`` if ``sub_agents`` is empty, since composite
        agents require at least one child agent.
        """
        if not sub_agents:
            raise ValueError(f"{agent_cls.__name__} requires at least one sub-agent.")
        if run_override is None:
            before_cb = self._resolve_shell_before_callback()
            after_cb = self._resolve_shell_after_callback()
        else:
            before_cb = cast(BeforeAgentCallback | None, self._resolve_callback("before_agent_callback"))
            after_cb = cast(AfterAgentCallback | None, self._resolve_callback("after_agent_callback"))
        kwargs: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "sub_agents": sub_agents,
            "before_agent_callback": before_cb,
            "after_agent_callback": after_cb,
        }
        return self._instantiate_with_optional_run_override(agent_cls, kwargs, run_override)

    def _build_loop(self, sub_agents, run_override):
        """
        Build a LoopAgent from sub-agents and optional run override.

        Parameters
        ----------
        sub_agents : list
            Child agents to pass to the LoopAgent constructor.
        run_override : callable or None
            Optional run_async_impl override.

        Returns
        -------
        LoopAgent
            Constructed LoopAgent instance.

        Notes
        -----
        Raises ``ValueError`` if ``sub_agents`` is empty. Passes
        ``max_iterations`` to the LoopAgent constructor only when set.
        """
        if not sub_agents:
            raise ValueError("LoopAgent requires at least one sub-agent.")
        if run_override is None:
            before_cb = self._resolve_shell_before_callback()
            after_cb = self._resolve_shell_after_callback()
        else:
            before_cb = cast(BeforeAgentCallback | None, self._resolve_callback("before_agent_callback"))
            after_cb = cast(AfterAgentCallback | None, self._resolve_callback("after_agent_callback"))
        kwargs: dict[str, Any] = {
            "name": self.name, "description": self.description,
            "sub_agents": sub_agents,
            "before_agent_callback": before_cb, "after_agent_callback": after_cb,
        }
        if self.max_iterations is not None:
            kwargs["max_iterations"] = self.max_iterations
        return self._instantiate_with_optional_run_override(LoopAgent, kwargs, run_override)

    def _resolve_shell_before_callback(self):
        """
        Select the before-callback for shell composite agents.

        Returns
        -------
        BeforeAgentCallback or None
            The before_model_callback if overridden, else before_agent_callback.

        Notes
        -----
        Composite agents (sequential/parallel/loop) have no model callbacks,
        so the before_model_callback override is promoted to before_agent_callback
        position when present.
        """
        if self._has_callback_override("before_model_callback"):
            return cast(BeforeAgentCallback | None, self._resolve_callback("before_model_callback"))
        return cast(BeforeAgentCallback | None, self._resolve_callback("before_agent_callback"))

    def _resolve_shell_after_callback(self):
        """
        Select the after-callback for shell composite agents.

        Returns
        -------
        AfterAgentCallback or None
            The after_model_callback if overridden, else after_agent_callback.

        Notes
        -----
        Same promotion logic as ``_resolve_shell_before_callback`` but for
        the after-phase callback.
        """
        if self._has_callback_override("after_model_callback"):
            return cast(AfterAgentCallback | None, self._resolve_callback("after_model_callback"))
        return cast(AfterAgentCallback | None, self._resolve_callback("after_agent_callback"))

    def _resolve_callback(self, callback_name: str) -> Any:
        """
        Resolve the effective callback value for a named callback attribute.

        Parameters
        ----------
        callback_name : str
            Name of the callback attribute to resolve.

        Returns
        -------
        Any
            The resolved callback: method override, explicit attribute, or class default.

        Notes
        -----
        Resolution priority: (1) method override on subclass, (2) explicit
        attribute assignment, (3) class-level default from BaseAgent.
        """
        method_override = self._resolve_callback_method_override(callback_name)
        if method_override is not None:
            return method_override
        explicit_attr = self._resolve_explicit_callback_attribute(callback_name)
        if explicit_attr is not None:
            return explicit_attr
        return BaseAgent.__dict__[callback_name]

    def _resolve_callback_method_override(self, callback_name: str) -> Any | None:
        """
        Detect and return a method-level callback override from a subclass.

        Parameters
        ----------
        callback_name : str
            Name of the callback to check for method overrides.

        Returns
        -------
        Any or None
            Bound method if a method override is found, else ``None``.

        Notes
        -----
        Traverses the MRO stopping at BaseAgent. Only returns a result if the
        attribute is an actual function (not a re-assigned callable value).
        """
        for cls in type(self).mro():
            if cls is BaseAgent:
                break
            if callback_name not in cls.__dict__:
                continue
            raw_value = cls.__dict__[callback_name]
            if inspect.isfunction(raw_value) and raw_value.__name__ == callback_name:
                return raw_value.__get__(self, type(self))
            return None
        return None

    def _resolve_explicit_callback_attribute(self, callback_name: str) -> Any | None:
        """
        Detect and return an explicitly assigned callback attribute value.

        Parameters
        ----------
        callback_name : str
            Name of the callback attribute to check.

        Returns
        -------
        Any or None
            The explicitly assigned value, or ``None`` if not found.

        Notes
        -----
        Checks both instance ``__dict__`` and class ``__dict__`` in MRO order.
        Returns ``None`` if the value is a function (which is handled by the
        method override resolver instead).
        """
        if callback_name in self.__dict__:
            return self.__dict__[callback_name]
        for cls in type(self).mro():
            if cls is BaseAgent:
                break
            if callback_name not in cls.__dict__:
                continue
            raw_value = cls.__dict__[callback_name]
            if inspect.isfunction(raw_value) and raw_value.__name__ == callback_name:
                return None
            return raw_value
        return None

    def _has_callback_override(self, callback_name: str) -> bool:
        """
        Return True if a callback has been overridden from the BaseAgent default.

        Parameters
        ----------
        callback_name : str
            Name of the callback to check.

        Returns
        -------
        bool
            ``True`` if a method or attribute override exists for this callback.

        Notes
        -----
        Used by composite agent builders to decide which callbacks to promote.
        """
        return (
            self._resolve_callback_method_override(callback_name) is not None
            or self._resolve_explicit_callback_attribute(callback_name) is not None
        )

    def _resolve_run_override(self) -> RunAsyncOverride | None:
        """
        Detect whether a run_async_impl or _run_async_impl override is defined.

        Returns
        -------
        RunAsyncOverride or None
            The bound override method, or ``None`` if no override is present.

        Notes
        -----
        Prefers the public ``run_async_impl`` over the private ``_run_async_impl``.
        Warns once if both are defined on the same class.
        """
        cls = type(self)
        has_public = (
            cls.__dict__.get("run_async_impl") is not None
            and cls.__dict__.get("run_async_impl") is not BaseAgent.run_async_impl
        )
        has_private = (
            cls.__dict__.get("_run_async_impl") is not None
            and cls.__dict__.get("_run_async_impl") is not BaseAgent._run_async_impl
        )
        if not has_public and not has_private:
            return None
        if has_public and has_private:
            self._warn_run_override_conflict_once(cls)
        selected_name = "run_async_impl" if has_public else "_run_async_impl"
        selected_override = getattr(self, selected_name)
        if not callable(selected_override):
            raise TypeError(f"{cls.__name__}.{selected_name} must be callable.")
        return selected_override

    @staticmethod
    def _warn_run_override_conflict_once(agent_cls: type[BaseAgent]) -> None:
        """
        Emit a one-time warning when both run override methods are defined.

        Parameters
        ----------
        agent_cls : type
            The agent subclass that defines both override methods.

        Returns
        -------
        None

        Notes
        -----
        Uses a module-level set to ensure the warning is emitted at most once
        per class, avoiding repeated log spam.
        """
        if agent_cls in _WARNED_RUN_OVERRIDE_CONFLICT:
            return
        _WARNED_RUN_OVERRIDE_CONFLICT.add(agent_cls)
        logger.warning("Both run_async_impl and _run_async_impl defined on {}. Using run_async_impl.", agent_cls.__name__)

    def _instantiate_with_optional_run_override(self, agent_cls, kwargs, run_override):
        """
        Instantiate an ADK agent class, injecting a run_async_impl override if provided.

        Parameters
        ----------
        agent_cls : type
            The ADK agent class to instantiate.
        kwargs : dict
            Constructor keyword arguments.
        run_override : callable or None
            Optional async generator function to use as ``_run_async_impl``.

        Returns
        -------
        ADKAgentType
            The constructed ADK agent instance.

        Notes
        -----
        When ``run_override`` is provided, a dynamic subclass is created that
        wraps the override in ``_run_async_impl``. The subclass name includes
        the suffix ``"WithRunOverride"`` for clarity in logs and tracebacks.
        """
        if run_override is None:
            return agent_cls(**kwargs)
        run_impl = cast(RunAsyncOverride, run_override)

        class _RunOverrideAgent(agent_cls):
            async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
                async for event in run_impl(ctx):
                    yield event

        _RunOverrideAgent.__name__ = f"{agent_cls.__name__}WithRunOverride"
        return cast(agent_cls, _RunOverrideAgent(**kwargs))

    @staticmethod
    def _supported_agent_fields() -> set[str]:
        """
        Return the set of field names supported by the installed ADK Agent class.

        Returns
        -------
        set of str
            Set of supported constructor keyword names, or an empty set if the
            fields cannot be determined from the installed ADK version.

        Notes
        -----
        Introspects ``Agent.model_fields`` (Pydantic v2) or ``Agent.__fields__``
        (Pydantic v1) to build the supported field set for compatibility checking.
        """
        for attr_name in ("model_fields", "__fields__"):
            fields = getattr(Agent, attr_name, None)
            if isinstance(fields, dict) and fields:
                return set(fields.keys())
        return set()

    def _is_default_optional_error_callback(self, key: str, value: Any) -> bool:
        """
        Return True if the value matches the BaseAgent default for an error callback.

        Parameters
        ----------
        key : str
            Callback attribute name (e.g. ``"on_model_error_callback"``).
        value : Any
            The current value to compare against the default.

        Returns
        -------
        bool
            ``True`` if the value is the same as the class-level default.

        Notes
        -----
        Used to suppress spurious unsupported-kwarg warnings for error callbacks
        that are ``None`` by default and not supported by older ADK versions.
        """
        defaults = {
            "on_model_error_callback": BaseAgent.on_model_error_callback,
            "on_tool_error_callback": BaseAgent.on_tool_error_callback,
        }
        default_value = defaults.get(key)
        if default_value is None:
            return False
        return self._callbacks_match(value, default_value)

    @staticmethod
    def _callbacks_match(left: Any, right: Any) -> bool:
        """
        Return True if two callback values refer to the same underlying function.

        Parameters
        ----------
        left : Any
            First callback to compare.
        right : Any
            Second callback to compare.

        Returns
        -------
        bool
            ``True`` if both callbacks point to the same function, optionally
            with the same binding.

        Notes
        -----
        Handles bound methods by comparing both the underlying function
        (``__func__``) and the instance (``__self__``). Treats unbound
        functions as equal regardless of binding context.
        """
        if left is right:
            return True
        left_func = getattr(left, "__func__", left)
        right_func = getattr(right, "__func__", right)
        left_self = getattr(left, "__self__", None)
        right_self = getattr(right, "__self__", None)
        if left_func is right_func:
            if left_self is None or right_self is None:
                return True
            return left_self is right_self
        return False

    @staticmethod
    def _warn_unsupported_kwarg_once(key: str) -> None:
        """
        Emit a one-time warning for an unsupported Agent constructor kwarg.

        Parameters
        ----------
        key : str
            The unsupported keyword argument name.

        Returns
        -------
        None

        Notes
        -----
        Uses a module-level set to ensure each key is warned about at most once
        per process lifetime, preventing repeated log spam during hot reload.
        """
        if key in _WARNED_UNSUPPORTED_AGENT_KWARGS:
            return
        _WARNED_UNSUPPORTED_AGENT_KWARGS.add(key)
        logger.warning("Ignoring unsupported Agent kwarg `{}` for installed google-adk.", key)
