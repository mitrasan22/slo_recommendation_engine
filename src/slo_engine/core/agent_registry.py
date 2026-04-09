"""
Central agent registry mapping agent names to ADK agent instances.

Notes
-----
Provides a module-level dictionary backed by the ``AgentRegistry`` class.
Agents register themselves by name on initialisation and are retrieved by
name when the router needs to dispatch to a specific sub-agent.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from google.adk.agents import Agent, LoopAgent, ParallelAgent, SequentialAgent

    ADKAgentType = Agent | SequentialAgent | ParallelAgent | LoopAgent

logger = logger.bind(name=__name__)

_registry: dict[str, ADKAgentType] = {}


class AgentRegistry:
    """
    Class-level registry for ADK agent instances.

    Notes
    -----
    All methods are class methods operating on the module-level ``_registry``
    dict. Agents are stored by name and retrieved by name. Overwriting an
    existing registration logs a debug message but does not raise an error.
    """

    @classmethod
    def register(cls, name: str, agent: ADKAgentType) -> None:
        """
        Register an ADK agent under the given name.

        Parameters
        ----------
        name : str
            Unique string identifier for the agent.
        agent : ADKAgentType
            The built ADK agent instance to register.

        Returns
        -------
        None

        Notes
        -----
        If ``name`` is already registered, the existing entry is overwritten
        and a debug log message is emitted. No exception is raised.
        """
        if name in _registry:
            logger.debug("AgentRegistry: overwriting '{}' with a new instance.", name)
        _registry[name] = agent
        logger.debug("AgentRegistry: registered '{}'.", name)

    @classmethod
    def get(cls, name: str) -> ADKAgentType | None:
        """
        Retrieve a registered agent by name.

        Parameters
        ----------
        name : str
            Name of the agent to look up.

        Returns
        -------
        ADKAgentType or None
            The registered agent instance, or None if not found.

        Notes
        -----
        Returns None silently when the name is not in the registry. Callers
        are responsible for handling the None case.
        """
        return _registry.get(name)

    @classmethod
    def all(cls) -> dict[str, ADKAgentType]:
        """
        Return a shallow copy of all registered agents.

        Returns
        -------
        dict of str to ADKAgentType
            Mapping of agent name to agent instance for all registered agents.

        Notes
        -----
        Returns a copy of the internal registry dict so that callers cannot
        mutate the registry directly through the returned dict.
        """
        return dict(_registry)
