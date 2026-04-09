"""
Platform-agnostic service catalog adapter.

Notes
-----
Normalises any developer portal's entity format into the engine's internal
ServiceGraph format consumed by ``ingest_service_dependencies``.

Output schema per service entry::

    {
        "service": str,
        "tier": str | None,
        "p99_latency_ms": float | None,
        "depends_on": [
            {"name": str, "dep_type": str, "p99_latency_ms": float | None}
        ]
    }
"""
from __future__ import annotations

from loguru import logger

logger = logger.bind(name=__name__)


def detect_platform(payload: dict) -> str:
    """
    Infer the developer portal platform from a catalog payload structure.

    Parameters
    ----------
    payload : dict
        Raw catalog payload with an ``entities`` list.

    Returns
    -------
    str
        Detected platform string: ``"backstage"``, ``"port"``, ``"cortex"``,
        or ``"generic"``.

    Notes
    -----
    Detection is based on the presence of distinguishing keys in the first
    entity of the ``entities`` list:
    - Backstage: has ``apiVersion``, ``kind``, and ``metadata`` keys.
    - Port: has ``blueprint`` or ``identifier`` keys.
    - Cortex: has ``customData`` or both ``tag`` and ``dependencies`` keys.
    - Generic: no distinguishing keys found.
    """
    entities = payload.get("entities", [])
    if not entities:
        return "generic"

    first = entities[0] if isinstance(entities, list) else payload

    if "apiVersion" in first and "kind" in first and "metadata" in first:
        return "backstage"

    if "blueprint" in first or "identifier" in first:
        return "port"

    if "customData" in first or ("tag" in first and "dependencies" in first):
        return "cortex"

    return "generic"


def from_backstage(entities: list[dict]) -> list[dict]:
    """
    Map Backstage Component entities to the ServiceGraph format.

    Parameters
    ----------
    entities : list of dict
        List of Backstage entity objects.

    Returns
    -------
    list of dict
        Normalised service graph entries.

    Notes
    -----
    Field mapping:
    - ``entity.metadata.name`` -> service name.
    - ``entity.metadata.annotations["slo/tier"]`` -> tier.
    - ``entity.metadata.annotations["slo/p99-latency-ms"]`` -> p99_latency_ms.
    - ``entity.spec.dependsOn`` -> depends_on list; ``"component:"`` prefix stripped.
    All dependency types default to ``"synchronous"`` since Backstage does not
    natively encode dependency type semantics.
    """
    result: list[dict] = []
    for entity in entities:
        if not isinstance(entity, dict):
            continue
        meta = entity.get("metadata", {})
        spec = entity.get("spec", {})
        annotations = meta.get("annotations", {})

        name = meta.get("name", "")
        if not name:
            logger.warning("Backstage entity missing metadata.name, skipping")
            continue

        tier = annotations.get("slo/tier") or None
        raw_lat = annotations.get("slo/p99-latency-ms")
        p99 = float(raw_lat) if raw_lat is not None else None

        raw_deps = spec.get("dependsOn", []) or []
        depends_on: list[dict] = []
        for dep in raw_deps:
            dep_name = str(dep)
            if ":" in dep_name:
                dep_name = dep_name.split(":", 1)[1]
            depends_on.append({"name": dep_name, "dep_type": "synchronous", "p99_latency_ms": None})

        entry: dict = {"service": name, "depends_on": depends_on}
        if tier is not None:
            entry["tier"] = tier
        if p99 is not None:
            entry["p99_latency_ms"] = p99

        result.append(entry)

    logger.info("from_backstage: normalized {} entities", len(result))
    return result


def from_port(entities: list[dict]) -> list[dict]:
    """
    Map Port blueprint entities to the ServiceGraph format.

    Parameters
    ----------
    entities : list of dict
        List of Port entity objects.

    Returns
    -------
    list of dict
        Normalised service graph entries.

    Notes
    -----
    Field mapping:
    - ``entity.identifier`` or ``entity.title`` -> service name.
    - ``entity.properties.tier`` -> tier.
    - ``entity.properties.p99LatencyMs`` -> p99_latency_ms.
    - ``entity.relations`` -> depends_on list; relation ``type`` is used as dep_type.
    Each relation entry may be a dict with ``target`` and ``type`` keys or a bare
    string, in which case dep_type defaults to ``"synchronous"``.
    """
    result: list[dict] = []
    for entity in entities:
        if not isinstance(entity, dict):
            continue

        name = entity.get("identifier") or entity.get("title", "")
        if not name:
            logger.warning("Port entity missing identifier/title, skipping")
            continue

        props = entity.get("properties", {}) or {}
        tier = props.get("tier") or None
        raw_lat = props.get("p99LatencyMs")
        p99 = float(raw_lat) if raw_lat is not None else None

        raw_rels = entity.get("relations", []) or []
        depends_on: list[dict] = []
        for rel in raw_rels:
            if isinstance(rel, dict):
                dep_name = rel.get("target", "")
                dep_type = rel.get("type", "synchronous")
            else:
                dep_name = str(rel)
                dep_type = "synchronous"
            if dep_name:
                depends_on.append({"name": dep_name, "dep_type": dep_type, "p99_latency_ms": None})

        entry: dict = {"service": name, "depends_on": depends_on}
        if tier is not None:
            entry["tier"] = tier
        if p99 is not None:
            entry["p99_latency_ms"] = p99

        result.append(entry)

    logger.info("from_port: normalized {} entities", len(result))
    return result


def from_cortex(entities: list[dict]) -> list[dict]:
    """
    Map Cortex service catalog entries to the ServiceGraph format.

    Parameters
    ----------
    entities : list of dict
        List of Cortex entity objects.

    Returns
    -------
    list of dict
        Normalised service graph entries.

    Notes
    -----
    Field mapping:
    - ``entity.tag`` or ``entity.name`` -> service name.
    - ``entity.customData.tier`` -> tier.
    - ``entity.customData.p99LatencyMs`` -> p99_latency_ms.
    - ``entity.dependencies`` -> depends_on list; entries may be dicts with
      ``tag``/``name`` and ``type`` keys, or bare strings.
    """
    result: list[dict] = []
    for entity in entities:
        if not isinstance(entity, dict):
            continue

        name = entity.get("tag") or entity.get("name", "")
        if not name:
            logger.warning("Cortex entity missing tag/name, skipping")
            continue

        custom = entity.get("customData", {}) or {}
        tier = custom.get("tier") or None
        raw_lat = custom.get("p99LatencyMs")
        p99 = float(raw_lat) if raw_lat is not None else None

        raw_deps = entity.get("dependencies", []) or []
        depends_on: list[dict] = []
        for dep in raw_deps:
            if isinstance(dep, dict):
                dep_name = dep.get("tag") or dep.get("name", "")
                dep_type = dep.get("type", "synchronous")
            else:
                dep_name = str(dep)
                dep_type = "synchronous"
            if dep_name:
                depends_on.append({"name": dep_name, "dep_type": dep_type, "p99_latency_ms": None})

        entry: dict = {"service": name, "depends_on": depends_on}
        if tier is not None:
            entry["tier"] = tier
        if p99 is not None:
            entry["p99_latency_ms"] = p99

        result.append(entry)

    logger.info("from_cortex: normalized {} entities", len(result))
    return result


def from_generic(entities: list[dict]) -> list[dict]:
    """
    Pass-through adapter for already-normalised data or custom platforms.

    Parameters
    ----------
    entities : list of dict
        List of entity objects that should already conform to the ServiceGraph
        format with at minimum a ``service`` key and a ``depends_on`` list.

    Returns
    -------
    list of dict
        Validated and pass-through entity list with malformed entries dropped.

    Notes
    -----
    Entries missing the ``service`` key are dropped with a warning. Entries
    with a non-list ``depends_on`` value have it replaced with an empty list
    and a warning is emitted. All other fields are passed through unchanged.
    """
    result: list[dict] = []
    for entity in entities:
        if not isinstance(entity, dict):
            logger.warning("from_generic: skipping non-dict entity")
            continue
        if "service" not in entity:
            logger.warning("from_generic: entity missing 'service' key, skipping: {}", entity)
            continue
        if not isinstance(entity.get("depends_on", []), list):
            logger.warning("from_generic: 'depends_on' is not a list for '{}', defaulting to []", entity["service"])
            entity = {**entity, "depends_on": []}
        result.append(entity)

    logger.info("from_generic: pass-through {} entities", len(result))
    return result
