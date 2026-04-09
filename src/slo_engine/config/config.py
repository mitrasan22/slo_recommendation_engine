"""
Project configuration — Config(Dynaconf) subclass with YAML loader.

Notes
-----
Loads environment variables from the project ``.env`` file before initialising
Dynaconf. Merges ``dev.yaml``, ``prod.yaml``, and ``.secrets.toml`` from the
config directory. Environment is selected via the ``SLO_ENV`` variable.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Union

from dotenv import load_dotenv
from dynaconf import Dynaconf
from loguru import logger

_ENV_FILE = Path(__file__).resolve().parents[3] / ".env"
load_dotenv(dotenv_path=_ENV_FILE, override=False)

logger = logger.bind(module=__name__)

CONFIG_DIR = Path(__file__).parent.resolve()
ROOT_DIR   = CONFIG_DIR.parents[3]


class Config(Dynaconf):
    """
    Dynaconf subclass providing typed attribute access and path resolution.

    Attributes
    ----------
    ROOT_DIR : Path
        Resolved path to the ``slo_recommendation_engine`` project root.

    Notes
    -----
    Delegates all configuration reads to the Dynaconf parent class.
    Environment files are loaded in order: ``dev.yaml``, ``prod.yaml``,
    ``.secrets.toml``. The ``SLO_`` prefix is used for environment variable
    overrides. The ``SLO_ENV`` variable switches between environments.
    """

    def __init__(
        self,
        settings_files: list = [],
        env_switcher: str | None = None,
        environments: bool = True,
        merge_enabled: bool = True,
        envvar_prefix: str = "SLO",
        load_dotenv: bool = True,
        ROOT_DIR: Path = ROOT_DIR,
        **kwargs,
    ):
        """
        Initialise the Config instance with Dynaconf settings.

        Parameters
        ----------
        settings_files : list, optional
            List of settings file paths to load in order.
        env_switcher : str or None, optional
            Environment variable name used to switch environments.
        environments : bool, optional
            Whether to enable Dynaconf multi-environment support. Defaults to True.
        merge_enabled : bool, optional
            Whether to enable Dynaconf deep merge for nested settings. Defaults to True.
        envvar_prefix : str, optional
            Prefix for environment variable overrides. Defaults to ``"SLO"``.
        load_dotenv : bool, optional
            Whether Dynaconf should load ``.env`` files. Defaults to True.
        ROOT_DIR : Path, optional
            Project root path used as ``ROOT_PATH_FOR_DYNACONF``.
        **kwargs : dict
            Additional keyword arguments forwarded to Dynaconf.

        Returns
        -------
        None

        Notes
        -----
        Sets ``ROOT_PATH_FOR_DYNACONF`` to the string form of ``ROOT_DIR``
        so that relative file paths in YAML configs resolve correctly.
        """
        super().__init__(
            settings_files=settings_files,
            ROOT_PATH_FOR_DYNACONF=str(ROOT_DIR),
            environments=environments,
            env_switcher=env_switcher,
            merge_enabled=merge_enabled,
            envvar_prefix=envvar_prefix,
            load_dotenv=load_dotenv,
            **kwargs,
        )
        self.ROOT_DIR: Path = ROOT_DIR
        self.__kwargs = kwargs

    def __getattr__(self, name: str) -> Union["Config", List, Dict, str]:
        """
        Proxy attribute access to the Dynaconf parent.

        Parameters
        ----------
        name : str
            Name of the configuration attribute to retrieve.

        Returns
        -------
        Config or list or dict or str
            Configuration value from Dynaconf.

        Notes
        -----
        Delegates directly to ``Dynaconf.__getattr__`` to preserve all
        Dynaconf behaviour including lazy loading and type coercion.
        """
        return super().__getattr__(name)


config = Config(
    settings_files=[
        str(CONFIG_DIR / "dev.yaml"),
        str(CONFIG_DIR / "prod.yaml"),
        str(ROOT_DIR / ".secrets.toml"),
    ],
    env_switcher="SLO_ENV",
    environments=True,
    merge_enabled=True,
    ROOT_DIR=ROOT_DIR,
)
