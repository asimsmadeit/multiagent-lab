"""Lineage queries and W3C-PROV-inspired export (Plan 2).

Given validated event envelopes, this module answers "where did this event
come from and what did it produce": transitive parents, transitive
descendants, sibling events of the same model call, reachable activation
artifacts, and the scenario/config/label events in that closure. The export
is PROV-*inspired* — entities, activities, agents, and a small relation
vocabulary rendered as deterministic JSON — not a complete W3C PROV-JSON
serialization.

The module is purely in-memory over :class:`EventEnvelope` values; file IO
and stream validation belong to :mod:`interpretability.events.reader` and
:mod:`interpretability.events.replay`, which should be used to obtain
validated envelopes first.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from interpretability.events.schema import EventEnvelope

PROV_DIALECT = "esr-events-prov/1.0"

_SCENARIO_EVENT_TYPE = "ScenarioInstantiated"
_CONFIG_EVENT_TYPE = "RunConfigFrozen"
_LABEL_EVENT_TYPE = "BehaviorLabeled"


class ProvenanceError(RuntimeError):
    """Base class for lineage-query failures."""


class UnknownEventError(ProvenanceError, ValueError):
    """The queried event ID is not present in the supplied stream."""


class DuplicateEventIdError(ProvenanceError, ValueError):
    """The supplied stream contains two envelopes with one event ID."""


class LineageCycleError(ProvenanceError):
    """Parent references form a cycle, so lineage is not well defined."""


@dataclass(frozen=True)
class LineageTrace:
    """Deterministic lineage closure around one target event."""

    target_event_id: str
    ancestor_event_ids: tuple[str, ...]
    descendant_event_ids: tuple[str, ...]
    same_call_event_ids: tuple[str, ...]
    artifact_hashes: tuple[str, ...]
    scenario_event_ids: tuple[str, ...]
    config_event_ids: tuple[str, ...]
    label_event_ids: tuple[str, ...]
    actor_ids: tuple[str, ...]
    missing_parent_ids: tuple[str, ...]

    def closure_event_ids(self) -> tuple[str, ...]:
        """Target plus ancestors, descendants, and same-call siblings."""
        ordered: dict[str, None] = {self.target_event_id: None}
        for event_id in (
            *self.ancestor_event_ids,
            *self.descendant_event_ids,
            *self.same_call_event_ids,
        ):
            ordered.setdefault(event_id, None)
        return tuple(ordered)


def _semantic_order_key(envelope: EventEnvelope) -> tuple[str, int, str]:
    """Order on envelope content only, so input order never matters."""
    return (envelope.trial_id or "", envelope.sequence_num, envelope.event_id)


@dataclass(frozen=True)
class _LineageIndex:
    by_id: Mapping[str, EventEnvelope]
    children: Mapping[str, tuple[str, ...]]

    @classmethod
    def build(cls, events: Iterable[EventEnvelope]) -> "_LineageIndex":
        by_id: dict[str, EventEnvelope] = {}
        for envelope in events:
            if not isinstance(envelope, EventEnvelope):
                raise ProvenanceError(
                    "lineage queries require EventEnvelope values"
                )
            if envelope.event_id in by_id:
                raise DuplicateEventIdError(
                    f"duplicate event_id in stream: {envelope.event_id}"
                )
            by_id[envelope.event_id] = envelope
        children: dict[str, list[str]] = {}
        for envelope in sorted(by_id.values(), key=_semantic_order_key):
            for parent_id in envelope.parent_event_ids:
                children.setdefault(parent_id, []).append(envelope.event_id)
        return cls(
            by_id=by_id,
            children={
                parent: tuple(child_ids)
                for parent, child_ids in children.items()
            },
        )


def _require_acyclic(index: _LineageIndex) -> None:
    """Fail closed on any parent-reference cycle in the stream.

    The hash-chained writer should make cycles impossible, but lineage over
    a corrupt or hand-assembled stream must error rather than silently
    truncate. Kahn's algorithm over in-stream parent edges; references to
    absent parents are gaps, not edges.
    """
    in_degree = {event_id: 0 for event_id in index.by_id}
    for envelope in index.by_id.values():
        for parent_id in envelope.parent_event_ids:
            if parent_id in index.by_id:
                in_degree[envelope.event_id] += 1
    frontier = [eid for eid, degree in in_degree.items() if degree == 0]
    resolved = 0
    while frontier:
        event_id = frontier.pop()
        resolved += 1
        for child_id in index.children.get(event_id, ()):
            in_degree[child_id] -= 1
            if in_degree[child_id] == 0:
                frontier.append(child_id)
    if resolved != len(index.by_id):
        cyclic = sorted(
            eid for eid, degree in in_degree.items() if degree > 0
        )
        raise LineageCycleError(
            "parent references form a cycle involving: " + ", ".join(cyclic)
        )


def _walk(
    start_id: str,
    neighbours_of,
    *,
    index: _LineageIndex,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Breadth-first transitive closure over an acyclic reference graph.

    Returns ``(reached_ids, missing_ids)`` where ``reached_ids`` is ordered
    by (depth, semantic key) and ``missing_ids`` are referenced-but-absent
    neighbours.
    """
    reached: dict[str, None] = {}
    missing: dict[str, None] = {}
    frontier = [start_id]
    seen = {start_id}
    while frontier:
        next_layer: list[str] = []
        for event_id in frontier:
            for neighbour_id in neighbours_of(event_id):
                if neighbour_id in seen:
                    continue
                seen.add(neighbour_id)
                if neighbour_id not in index.by_id:
                    missing[neighbour_id] = None
                    continue
                reached[neighbour_id] = None
                next_layer.append(neighbour_id)
        next_layer.sort(key=lambda eid: _semantic_order_key(index.by_id[eid]))
        frontier = next_layer
    return tuple(reached), tuple(missing)


def trace_event(
    events: Iterable[EventEnvelope],
    event_id: str,
) -> LineageTrace:
    """Return the deterministic lineage closure around ``event_id``."""
    index = _LineageIndex.build(events)
    target = index.by_id.get(event_id)
    if target is None:
        raise UnknownEventError(f"unknown event_id: {event_id}")
    _require_acyclic(index)

    def parents_of(eid: str) -> tuple[str, ...]:
        envelope = index.by_id.get(eid)
        return envelope.parent_event_ids if envelope is not None else ()

    def children_of(eid: str) -> tuple[str, ...]:
        return index.children.get(eid, ())

    ancestors, missing_parents = _walk(event_id, parents_of, index=index)
    descendants, _ = _walk(event_id, children_of, index=index)

    same_call: tuple[str, ...] = ()
    if target.model_call_id is not None:
        same_call = tuple(
            envelope.event_id
            for envelope in sorted(
                index.by_id.values(), key=_semantic_order_key
            )
            if envelope.model_call_id == target.model_call_id
            and envelope.event_id != event_id
        )

    closure_ids: dict[str, None] = {event_id: None}
    for reached in (ancestors, descendants, same_call):
        for reached_id in reached:
            closure_ids.setdefault(reached_id, None)
    closure = [index.by_id[eid] for eid in closure_ids]

    artifact_hashes = tuple(
        sorted(
            {
                ref.artifact_hash
                for envelope in closure
                for ref in envelope.artifact_refs
            }
        )
    )

    def _of_type(event_type: str) -> tuple[str, ...]:
        return tuple(
            envelope.event_id
            for envelope in sorted(closure, key=_semantic_order_key)
            if envelope.event_type == event_type
        )

    actor_ids = tuple(
        sorted(
            {
                envelope.actor_id
                for envelope in closure
                if envelope.actor_id is not None
            }
        )
    )
    return LineageTrace(
        target_event_id=event_id,
        ancestor_event_ids=ancestors,
        descendant_event_ids=descendants,
        same_call_event_ids=same_call,
        artifact_hashes=artifact_hashes,
        scenario_event_ids=_of_type(_SCENARIO_EVENT_TYPE),
        config_event_ids=_of_type(_CONFIG_EVENT_TYPE),
        label_event_ids=_of_type(_LABEL_EVENT_TYPE),
        actor_ids=actor_ids,
        missing_parent_ids=tuple(sorted(missing_parents)),
    )


def prov_document(
    events: Iterable[EventEnvelope],
    event_id: str,
) -> dict[str, object]:
    """Export the lineage closure as a PROV-inspired JSON-compatible dict.

    Every closure event is an *activity*; activation artifacts, scenario
    instances, and frozen configs additionally appear as *entities*; actors
    are *agents*. Relations use ``wasInformedBy`` (event -> parent event),
    ``wasGeneratedBy`` (artifact -> capturing event), and
    ``wasAssociatedWith`` (event -> actor). All collections are sorted, so
    the document is byte-stable for one stream regardless of input order.
    """
    materialized = list(events)
    trace = trace_event(materialized, event_id)
    index = _LineageIndex.build(materialized)
    closure = {
        eid: index.by_id[eid] for eid in trace.closure_event_ids()
    }

    entities: dict[str, dict[str, object]] = {}
    for artifact_hash in trace.artifact_hashes:
        entities[f"artifact:{artifact_hash}"] = {
            "prov:type": "activation_artifact"
        }
    for scenario_id in trace.scenario_event_ids:
        entities[f"event:{scenario_id}"] = {"prov:type": "scenario_instance"}
    for config_id in trace.config_event_ids:
        entities[f"event:{config_id}"] = {"prov:type": "frozen_config"}

    activities = {
        f"event:{eid}": {
            "prov:type": envelope.event_type,
            "run_id": envelope.run_id,
            "trial_id": envelope.trial_id,
            "model_call_id": envelope.model_call_id,
        }
        for eid, envelope in sorted(closure.items())
    }
    agents = {
        f"actor:{actor_id}": {"prov:type": "agent"}
        for actor_id in trace.actor_ids
    }

    was_informed_by = sorted(
        [f"event:{envelope.event_id}", f"event:{parent_id}"]
        for envelope in closure.values()
        for parent_id in envelope.parent_event_ids
        if parent_id in closure
    )
    was_generated_by = sorted(
        [f"artifact:{ref.artifact_hash}", f"event:{envelope.event_id}"]
        for envelope in closure.values()
        for ref in envelope.artifact_refs
    )
    was_associated_with = sorted(
        [f"event:{envelope.event_id}", f"actor:{envelope.actor_id}"]
        for envelope in closure.values()
        if envelope.actor_id is not None
    )

    return {
        "prov_dialect": PROV_DIALECT,
        "target_event_id": trace.target_event_id,
        "missing_parent_ids": list(trace.missing_parent_ids),
        "entity": dict(sorted(entities.items())),
        "activity": activities,
        "agent": agents,
        "wasInformedBy": was_informed_by,
        "wasGeneratedBy": was_generated_by,
        "wasAssociatedWith": was_associated_with,
    }


__all__ = [
    "PROV_DIALECT",
    "DuplicateEventIdError",
    "LineageCycleError",
    "LineageTrace",
    "ProvenanceError",
    "UnknownEventError",
    "prov_document",
    "trace_event",
]
