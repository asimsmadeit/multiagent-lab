"""Permanent contracts for the explicit, versioned scenario registry."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
import importlib
import json
import os
from pathlib import Path
import random
from threading import Barrier
from typing import Any, Callable
import uuid

import pytest

import interpretability.scenarios.registry as registry_module
from interpretability.scenarios.compiler import compile_scenario
from interpretability.scenarios.registry import (
    DuplicateScenarioError,
    ExactVersionRequiredError,
    MAX_SCENARIO_SPEC_BYTES,
    REGISTRY_MANIFEST_VERSION,
    RegistryEntry,
    ScenarioCollisionError,
    ScenarioLoadError,
    ScenarioRegistry,
    ScenarioRegistryError,
    UnknownScenarioError,
    UnknownScenarioVersionError,
    UnsafeScenarioPathError,
    compile_registered,
    release_filename_for,
)
from interpretability.scenarios.schema import (
    ActionDefinition,
    ActionKind,
    AgreementState,
    BehaviorSubtype,
    BehaviorTargetDefinition,
    ConditionDefinition,
    ExtractorReference,
    FactDefinition,
    FactSamplingDefinition,
    FactSamplingKind,
    FactValueType,
    IncentiveCondition,
    OutcomeDefinition,
    PromptKind,
    PromptTemplate,
    RoleDefinition,
    RoleKind,
    RuleReference,
    ScenarioMetadata,
    ScenarioSpec,
    Visibility,
    canonical_json,
    canonical_sha256,
)


ExceptionType = type[BaseException] | tuple[type[BaseException], ...]


def _registry_spec() -> ScenarioSpec:
    """Build a small, fully compilable registry fixture."""
    return ScenarioSpec(
        spec_version="1.0.0",
        metadata=ScenarioMetadata(
            scenario_id="registry_fixture",
            display_name="Registry fixture",
            description="Scenario registry contract fixture.",
            research_constructs=(BehaviorSubtype.FALSE_CLAIM,),
            tags=("registry",),
        ),
        roles=(
            RoleDefinition(
                role_id="actor",
                kind=RoleKind.ACTOR,
                description="Acting negotiator.",
            ),
            RoleDefinition(
                role_id="counterpart",
                kind=RoleKind.COUNTERPART,
                description="Negotiating counterpart.",
            ),
        ),
        facts=(
            FactDefinition(
                fact_id="amount",
                fact_version="fact/1",
                value_type=FactValueType.INTEGER,
                visibility=Visibility.PUBLIC,
                description="Public amount.",
                sampling=FactSamplingDefinition(
                    kind=FactSamplingKind.FIXED,
                    fixed_value=10,
                ),
            ),
        ),
        conditions=(
            ConditionDefinition(
                condition=IncentiveCondition.MINIMAL,
                description="Minimal incentive condition.",
            ),
        ),
        prompt_templates=(
            PromptTemplate(
                template_id="actor.initial",
                template_version="prompt/1",
                role_id="actor",
                condition=IncentiveCondition.MINIMAL,
                kind=PromptKind.INITIAL,
                template="Offer {amount} in {trial_id}.",
            ),
            PromptTemplate(
                template_id="counterpart.reply",
                template_version="prompt/1",
                role_id="counterpart",
                condition=IncentiveCondition.MINIMAL,
                kind=PromptKind.COUNTERPART,
                template="Reply concerning {amount}.",
            ),
        ),
        action_space=(
            ActionDefinition(
                action_id="message",
                kind=ActionKind.MESSAGE,
                actor_role_ids=("actor", "counterpart"),
                description="Send a negotiation message.",
            ),
        ),
        extractors=(
            ExtractorReference(
                extractor_name="registry_parser",
                extractor_version="extractor/1",
                supported_action_kinds=(ActionKind.MESSAGE,),
                deterministic=True,
            ),
        ),
        rules=(
            RuleReference(
                rule_id="feasible",
                rule_version="rule/1",
                predicate_id="predicate.feasible",
                input_fact_ids=("amount",),
                description="Fixture feasibility rule.",
            ),
        ),
        behavior_targets=(
            BehaviorTargetDefinition(
                target_id="false_claim",
                subtype=BehaviorSubtype.FALSE_CLAIM,
                rule_ids=("feasible",),
                belief_dependent=True,
                default_severity=1.0,
            ),
        ),
        outcomes=(
            OutcomeDefinition(
                outcome_id="complete",
                rule_ids=("feasible",),
                agreement_state=AgreementState.UNKNOWN,
                utility_role_ids=("actor", "counterpart"),
                description="Completed fixture negotiation.",
            ),
        ),
    )


@pytest.fixture(scope="module")
def registry_spec() -> ScenarioSpec:
    """Return the canonical scenario used by registry contracts."""
    return _registry_spec()


def _copy_spec(
    spec: ScenarioSpec,
    *,
    scenario_id: str | None = None,
    spec_version: str | None = None,
    description: str | None = None,
) -> ScenarioSpec:
    metadata = ScenarioMetadata(
        scenario_id=scenario_id or spec.metadata.scenario_id,
        display_name=spec.metadata.display_name,
        description=description or spec.metadata.description,
        research_constructs=spec.metadata.research_constructs,
        tags=spec.metadata.tags,
    )
    return ScenarioSpec(
        spec_version=spec_version or spec.spec_version,
        metadata=metadata,
        roles=spec.roles,
        facts=spec.facts,
        conditions=spec.conditions,
        prompt_templates=spec.prompt_templates,
        intervention_points=spec.intervention_points,
        action_space=spec.action_space,
        extractors=spec.extractors,
        rules=spec.rules,
        behavior_targets=spec.behavior_targets,
        outcomes=spec.outcomes,
    )


def _assert_unchanged_failure(
    registry: ScenarioRegistry,
    exception: ExceptionType,
    operation: Callable[[], Any],
) -> None:
    before = registry.snapshot()
    with pytest.raises(exception):
        operation()
    assert registry.snapshot() == before


def _write_release(
    directory: Path,
    spec: ScenarioSpec,
    data: bytes | None = None,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / release_filename_for(
        spec.metadata.scenario_id,
        spec.spec_version,
    )
    path.write_bytes(data if data is not None else spec.canonical_json().encode())
    return path


def _symlink_or_skip(
    link: Path,
    target: Path,
    *,
    target_is_directory: bool,
) -> None:
    try:
        link.symlink_to(target, target_is_directory=target_is_directory)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")


def _temporary_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[str, Path]:
    package_name = f"registry_resources_{uuid.uuid4().hex}"
    import_root = tmp_path / "import-root"
    package_root = import_root / package_name
    package_root.mkdir(parents=True)
    (package_root / "__init__.py").write_text(
        '"""Temporary registry resource package."""\n',
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(import_root))
    importlib.invalidate_caches()
    return package_name, package_root


def test_exact_prerelease_build_lookup_and_explicit_defaults(
    registry_spec: ScenarioSpec,
) -> None:
    prerelease = _copy_spec(registry_spec, spec_version="2.0.0-rc.1")
    build = _copy_spec(registry_spec, spec_version="2.0.0+build.7")
    registry = ScenarioRegistry()
    registry.register_many((build, registry_spec, prerelease))

    _assert_unchanged_failure(
        registry,
        ExactVersionRequiredError,
        lambda: registry.get(registry_spec.metadata.scenario_id),
    )
    assert registry.get("registry_fixture", "1.0.0") == registry_spec
    assert registry.get("registry_fixture", "2.0.0-rc.1") == prerelease
    assert registry.get("registry_fixture", "2.0.0+build.7") == build

    registry.set_default("registry_fixture", "2.0.0-rc.1")
    assert registry.get("registry_fixture") == prerelease
    registry.set_default("registry_fixture", "1.0.0")
    assert registry.get("registry_fixture") == registry_spec
    assert tuple(entry.is_default for entry in registry.list()) == (
        True,
        False,
        False,
    )

    direct_default = ScenarioRegistry()
    direct_default.register(registry_spec, default=True)
    assert direct_default.get("registry_fixture") == registry_spec


@pytest.mark.parametrize("version", ["1", "1.0", "v1.0.0", "latest", ""])
def test_lookup_requires_an_exact_semantic_version(
    registry_spec: ScenarioSpec,
    version: str,
) -> None:
    registry = ScenarioRegistry()
    registry.register(registry_spec)
    _assert_unchanged_failure(
        registry,
        UnknownScenarioVersionError,
        lambda: registry.get("registry_fixture", version),
    )


def test_unknown_scenario_version_and_default_fail_without_mutation(
    registry_spec: ScenarioSpec,
) -> None:
    registry = ScenarioRegistry()
    registry.register(registry_spec)

    _assert_unchanged_failure(
        registry,
        UnknownScenarioError,
        lambda: registry.get("unknown_scenario", "1.0.0"),
    )
    _assert_unchanged_failure(
        registry,
        UnknownScenarioVersionError,
        lambda: registry.get("registry_fixture", "9.9.9"),
    )
    _assert_unchanged_failure(
        registry,
        UnknownScenarioVersionError,
        lambda: registry.set_default("registry_fixture", "9.9.9"),
    )
    _assert_unchanged_failure(
        registry,
        ScenarioRegistryError,
        lambda: registry.get("invalid scenario", "1.0.0"),
    )


def test_duplicate_and_identity_collision_fail_without_mutation(
    registry_spec: ScenarioSpec,
) -> None:
    registry = ScenarioRegistry()
    registry.register(registry_spec, source="first")

    _assert_unchanged_failure(
        registry,
        DuplicateScenarioError,
        lambda: registry.register(registry_spec, source="second"),
    )
    changed = _copy_spec(
        registry_spec,
        description="Different content under the same released identity.",
    )
    assert changed.spec_hash != registry_spec.spec_hash
    _assert_unchanged_failure(
        registry,
        ScenarioCollisionError,
        lambda: registry.register(changed),
    )


def test_content_hash_collision_branch_is_fail_closed(
    registry_spec: ScenarioSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Force the otherwise cryptographically infeasible hash-collision branch."""
    other = _copy_spec(
        registry_spec,
        scenario_id="other_registry_fixture",
        spec_version="3.0.0",
    )
    object.__setattr__(other, "spec_hash", registry_spec.spec_hash)
    monkeypatch.setattr(registry_module, "_validated_spec", lambda spec: spec)
    registry = ScenarioRegistry()
    registry.register(registry_spec)

    _assert_unchanged_failure(
        registry,
        ScenarioCollisionError,
        lambda: registry.register(other),
    )


def test_register_rejects_invalid_inputs_without_mutation(
    registry_spec: ScenarioSpec,
) -> None:
    registry = ScenarioRegistry()
    _assert_unchanged_failure(
        registry,
        TypeError,
        lambda: registry.register(object()),  # type: ignore[arg-type]
    )
    _assert_unchanged_failure(
        registry,
        TypeError,
        lambda: registry.register(registry_spec, source=""),
    )


def test_register_many_is_atomic_for_duplicate_and_default_failures(
    registry_spec: ScenarioSpec,
) -> None:
    version_two = _copy_spec(registry_spec, spec_version="2.0.0")
    scenario_id = registry_spec.metadata.scenario_id

    empty = ScenarioRegistry()
    _assert_unchanged_failure(
        empty,
        DuplicateScenarioError,
        lambda: empty.register_many((registry_spec, registry_spec)),
    )

    existing = ScenarioRegistry()
    existing.register(registry_spec)
    _assert_unchanged_failure(
        existing,
        DuplicateScenarioError,
        lambda: existing.register_many((version_two, registry_spec)),
    )
    _assert_unchanged_failure(
        existing,
        UnknownScenarioVersionError,
        lambda: existing.register_many(
            (version_two,),
            defaults=((scenario_id, "9.0.0"),),
        ),
    )

    defaulted = ScenarioRegistry()
    defaulted.register(registry_spec, default=True)
    _assert_unchanged_failure(
        defaulted,
        ScenarioRegistryError,
        lambda: defaulted.register_many(
            (version_two,),
            defaults=((scenario_id, version_two.spec_version),),
        ),
    )

    registered = existing.register_many(
        (version_two,),
        defaults=((scenario_id, version_two.spec_version),),
    )
    assert registered == (version_two,)
    assert existing.get(scenario_id) == version_two


def test_listing_snapshot_and_manifest_are_canonical_and_order_independent(
    registry_spec: ScenarioSpec,
) -> None:
    alpha_one = _copy_spec(registry_spec, scenario_id="alpha", spec_version="1.0.0")
    alpha_two = _copy_spec(registry_spec, scenario_id="alpha", spec_version="2.0.0")
    zeta = _copy_spec(registry_spec, scenario_id="zeta", spec_version="1.0.0")

    first = ScenarioRegistry()
    first.register(alpha_two, source="memory-b")
    first.register(zeta, source="memory-c")
    first.register(alpha_one, source="memory-a", default=True)
    second = ScenarioRegistry()
    second.register_many(
        (zeta, alpha_one, alpha_two),
        defaults=(("alpha", "1.0.0"),),
    )

    expected_entries = (
        RegistryEntry("alpha", "1.0.0", alpha_one.spec_hash, True),
        RegistryEntry("alpha", "2.0.0", alpha_two.spec_hash, False),
        RegistryEntry("zeta", "1.0.0", zeta.spec_hash, False),
    )
    assert first.list() == expected_entries
    assert first.list_specs() == expected_entries
    assert first.list("alpha") == expected_entries[:2]
    assert first.list_versions("alpha") == ("1.0.0", "2.0.0")
    assert len(first) == 3

    snapshot = first.snapshot()
    assert snapshot == second.snapshot()
    expected_hash = canonical_sha256(
        {
            "manifest_version": REGISTRY_MANIFEST_VERSION,
            "entries": [entry.to_dict() for entry in expected_entries],
        }
    )
    assert snapshot.manifest_hash == expected_hash
    assert snapshot.canonical_json() == canonical_json(
        {
            "manifest_version": REGISTRY_MANIFEST_VERSION,
            "entries": [entry.to_dict() for entry in expected_entries],
            "manifest_hash": expected_hash,
        }
    )

    assert first.list("unknown_scenario") == ()
    _assert_unchanged_failure(
        first,
        UnknownScenarioError,
        lambda: first.list_versions("unknown_scenario"),
    )


def test_snapshot_entries_and_manifest_are_immutable(
    registry_spec: ScenarioSpec,
) -> None:
    registry = ScenarioRegistry()
    registry.register(registry_spec, default=True)
    snapshot = registry.snapshot()
    entry = snapshot.entries[0]

    with pytest.raises(FrozenInstanceError):
        entry.is_default = False  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        snapshot.manifest_hash = "sha256:tampered"  # type: ignore[misc]
    copied = entry.to_dict()
    copied["spec_hash"] = "sha256:tampered"
    assert registry.snapshot() == snapshot
    assert snapshot.entries[0].spec_hash == registry_spec.spec_hash


def test_sixty_four_way_duplicate_registration_race_is_serialized(
    registry_spec: ScenarioSpec,
) -> None:
    registry = ScenarioRegistry()
    barrier = Barrier(64)

    def register_once(index: int) -> tuple[str, int]:
        barrier.wait(timeout=10)
        try:
            registry.register(registry_spec, source=f"thread-{index}")
        except DuplicateScenarioError:
            return "duplicate", index
        return "registered", index

    with ThreadPoolExecutor(max_workers=64) as pool:
        outcomes = tuple(pool.map(register_once, range(64)))

    assert sum(result == "registered" for result, _ in outcomes) == 1
    assert sum(result == "duplicate" for result, _ in outcomes) == 63
    assert len(registry) == 1
    assert registry.get("registry_fixture", "1.0.0") == registry_spec


def test_release_filename_is_encoded_and_bound_to_exact_identity(
    registry_spec: ScenarioSpec,
    tmp_path: Path,
) -> None:
    encoded = _copy_spec(
        registry_spec,
        scenario_id="vendor/path:offer",
        spec_version="1.2.3-rc.1+build.9",
    )
    expected = (
        "vendor%2Fpath%3Aoffer--1.2.3-rc.1%2Bbuild.9.scenario.json"
    )
    assert release_filename_for(
        encoded.metadata.scenario_id,
        encoded.spec_version,
    ) == expected

    wrong_name = tmp_path / "wrong.scenario.json"
    wrong_name.write_text(encoded.canonical_json(), encoding="utf-8")
    registry = ScenarioRegistry()
    _assert_unchanged_failure(
        registry,
        ScenarioLoadError,
        lambda: registry.load_file(wrong_name),
    )


@pytest.mark.parametrize(
    ("scenario_id", "version", "exception"),
    [
        ("invalid id", "1.0.0", ScenarioRegistryError),
        ("registry_fixture", "latest", UnknownScenarioVersionError),
        ("registry_fixture", "1.0", UnknownScenarioVersionError),
    ],
)
def test_release_filename_rejects_invalid_identity(
    scenario_id: str,
    version: str,
    exception: type[BaseException],
) -> None:
    with pytest.raises(exception):
        release_filename_for(scenario_id, version)


def test_canonical_absolute_and_relative_file_loads(
    registry_spec: ScenarioSpec,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "releases"
    path = _write_release(root, registry_spec)

    absolute = ScenarioRegistry()
    assert absolute.load_file(path, default=True) == registry_spec
    assert absolute.get("registry_fixture") == registry_spec

    rooted = ScenarioRegistry()
    assert rooted.load_file(path.name, root=root) == registry_spec

    relative = ScenarioRegistry()
    monkeypatch.chdir(root)
    assert relative.load_file(path.name) == registry_spec


def test_noncanonical_json_requires_an_explicit_compatibility_opt_out(
    registry_spec: ScenarioSpec,
    tmp_path: Path,
) -> None:
    pretty = json.dumps(
        json.loads(registry_spec.canonical_json()),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ).encode()
    path = _write_release(tmp_path, registry_spec, pretty)

    strict = ScenarioRegistry()
    _assert_unchanged_failure(
        strict,
        ScenarioLoadError,
        lambda: strict.load_file(path),
    )
    compatible = ScenarioRegistry()
    assert compatible.load_file(
        path,
        require_canonical_json=False,
    ) == registry_spec


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_spec_hash",
        "wrong_spec_hash",
        "wrong_schema_version",
        "tampered_nested_content",
        "extra_field",
    ],
)
def test_file_load_requires_complete_persisted_schema_and_hash_identity(
    registry_spec: ScenarioSpec,
    tmp_path: Path,
    mutation: str,
) -> None:
    payload = json.loads(registry_spec.canonical_json())
    if mutation == "missing_spec_hash":
        del payload["spec_hash"]
    elif mutation == "wrong_spec_hash":
        payload["spec_hash"] = "sha256:" + ("0" * 64)
    elif mutation == "wrong_schema_version":
        payload["schema_version"] = "scenario-dsl/999.0.0"
    elif mutation == "tampered_nested_content":
        payload["metadata"]["description"] = "Tampered after hashing."
    else:
        payload["unexpected"] = "forbidden"
    path = _write_release(
        tmp_path,
        registry_spec,
        canonical_json(payload).encode(),
    )
    registry = ScenarioRegistry()
    _assert_unchanged_failure(
        registry,
        ScenarioLoadError,
        lambda: registry.load_file(path),
    )


@pytest.mark.parametrize(
    ("name", "payload"),
    [
        ("duplicate-key", b'{"spec_hash":"first","spec_hash":"second"}'),
        ("nan", b'{"value":NaN}'),
        ("positive-infinity", b'{"value":Infinity}'),
        ("negative-infinity", b'{"value":-Infinity}'),
        ("overflow", b'{"value":1e999}'),
        ("invalid-utf8", b"\xff\xfe"),
        ("array", b"[]"),
        ("string", b'"scenario"'),
        ("number", b"42"),
        ("null", b"null"),
        ("malformed", b"{"),
    ],
)
def test_file_load_rejects_hostile_wire_payloads_without_mutation(
    registry_spec: ScenarioSpec,
    tmp_path: Path,
    name: str,
    payload: bytes,
) -> None:
    path = _write_release(tmp_path / name, registry_spec, payload)
    registry = ScenarioRegistry()
    _assert_unchanged_failure(
        registry,
        ScenarioLoadError,
        lambda: registry.load_file(path),
    )


def test_file_load_rejects_oversize_payload_before_json_decode(
    registry_spec: ScenarioSpec,
    tmp_path: Path,
) -> None:
    path = _write_release(
        tmp_path,
        registry_spec,
        b"x" * (MAX_SCENARIO_SPEC_BYTES + 1),
    )
    registry = ScenarioRegistry()
    _assert_unchanged_failure(
        registry,
        ScenarioLoadError,
        lambda: registry.load_file(path),
    )


def test_missing_directory_fifo_and_root_shape_are_rejected(
    registry_spec: ScenarioSpec,
    tmp_path: Path,
) -> None:
    registry = ScenarioRegistry()
    filename = release_filename_for("registry_fixture", "1.0.0")
    missing = tmp_path / "missing" / filename
    _assert_unchanged_failure(
        registry,
        UnsafeScenarioPathError,
        lambda: registry.load_file(missing),
    )

    directory = tmp_path / "directory" / filename
    directory.mkdir(parents=True)
    _assert_unchanged_failure(
        registry,
        UnsafeScenarioPathError,
        lambda: registry.load_file(directory),
    )

    root_file = _write_release(tmp_path / "root-file", registry_spec)
    _assert_unchanged_failure(
        registry,
        UnsafeScenarioPathError,
        lambda: registry.load_file(filename, root=root_file),
    )
    _assert_unchanged_failure(
        registry,
        UnsafeScenarioPathError,
        lambda: registry.load_file(filename, root=tmp_path / "absent-root"),
    )

    if hasattr(os, "mkfifo"):
        fifo = tmp_path / "fifo" / filename
        fifo.parent.mkdir()
        os.mkfifo(fifo)
        _assert_unchanged_failure(
            registry,
            UnsafeScenarioPathError,
            lambda: registry.load_file(fifo),
        )


def test_rooted_loader_rejects_absolute_and_parent_traversal_paths(
    registry_spec: ScenarioSpec,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "trusted"
    root.mkdir()
    outside = _write_release(tmp_path / "outside", registry_spec)
    registry = ScenarioRegistry()

    _assert_unchanged_failure(
        registry,
        UnsafeScenarioPathError,
        lambda: registry.load_file(outside, root=root),
    )
    _assert_unchanged_failure(
        registry,
        UnsafeScenarioPathError,
        lambda: registry.load_file(
            f"../outside/{outside.name}",
            root=root,
        ),
    )

    inside = root / "inside"
    inside.mkdir()
    monkeypatch.chdir(inside)
    _assert_unchanged_failure(
        registry,
        UnsafeScenarioPathError,
        lambda: registry.load_file(f"../../outside/{outside.name}"),
    )


def test_direct_root_and_descendant_symlinks_are_rejected(
    registry_spec: ScenarioSpec,
    tmp_path: Path,
) -> None:
    physical = tmp_path / "physical"
    target = _write_release(physical, registry_spec)
    registry = ScenarioRegistry()

    direct = tmp_path / "direct" / target.name
    direct.parent.mkdir()
    _symlink_or_skip(direct, target, target_is_directory=False)
    _assert_unchanged_failure(
        registry,
        UnsafeScenarioPathError,
        lambda: registry.load_file(direct),
    )

    root_link = tmp_path / "root-link"
    _symlink_or_skip(root_link, physical, target_is_directory=True)
    _assert_unchanged_failure(
        registry,
        UnsafeScenarioPathError,
        lambda: registry.load_file(target.name, root=root_link),
    )

    trusted = tmp_path / "trusted"
    trusted.mkdir()
    descendant_link = trusted / "linked-directory"
    _symlink_or_skip(descendant_link, physical, target_is_directory=True)
    _assert_unchanged_failure(
        registry,
        UnsafeScenarioPathError,
        lambda: registry.load_file(
            f"linked-directory/{target.name}",
            root=trusted,
        ),
    )


def test_resolved_root_accepts_a_trusted_ancestor_alias(
    registry_spec: ScenarioSpec,
    tmp_path: Path,
) -> None:
    """A platform-style ancestor alias is part of the declared root identity."""
    physical_parent = tmp_path / "physical-parent"
    physical_root = physical_parent / "released"
    _write_release(physical_root, registry_spec)
    alias = tmp_path / "platform-alias"
    _symlink_or_skip(alias, physical_parent, target_is_directory=True)
    declared_root = alias / "released"
    assert not declared_root.is_symlink()

    registry = ScenarioRegistry()
    loaded = registry.load_file(
        release_filename_for("registry_fixture", "1.0.0"),
        root=declared_root,
    )
    assert loaded == registry_spec


def test_file_batch_load_and_default_commit_atomically(
    registry_spec: ScenarioSpec,
    tmp_path: Path,
) -> None:
    version_two = _copy_spec(registry_spec, spec_version="2.0.0")
    root = tmp_path / "releases"
    first = _write_release(root, registry_spec)
    second = _write_release(root, version_two)
    registry = ScenarioRegistry()

    loaded = registry.load_files(
        (second.name, first.name),
        root=root,
        defaults=(("registry_fixture", "2.0.0"),),
    )
    assert loaded == (version_two, registry_spec)
    assert registry.get("registry_fixture") == version_two


def test_file_batch_decode_duplicate_and_default_failures_are_atomic(
    registry_spec: ScenarioSpec,
    tmp_path: Path,
) -> None:
    version_two = _copy_spec(registry_spec, spec_version="2.0.0")
    root = tmp_path / "releases"
    first = _write_release(root, registry_spec)
    second = _write_release(root, version_two)
    invalid = _write_release(tmp_path / "invalid", registry_spec, b"\xff")

    empty = ScenarioRegistry()
    _assert_unchanged_failure(
        empty,
        ScenarioLoadError,
        lambda: empty.load_files((first, invalid)),
    )
    _assert_unchanged_failure(
        empty,
        DuplicateScenarioError,
        lambda: empty.load_files((first, first)),
    )
    _assert_unchanged_failure(
        empty,
        UnknownScenarioVersionError,
        lambda: empty.load_files(
            (second,),
            defaults=(("registry_fixture", "9.0.0"),),
        ),
    )

    existing = ScenarioRegistry()
    existing.load_file(first)
    _assert_unchanged_failure(
        existing,
        DuplicateScenarioError,
        lambda: existing.load_files((second, first)),
    )


def test_package_resource_single_batch_and_defaults(
    registry_spec: ScenarioSpec,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    version_two = _copy_spec(registry_spec, spec_version="2.0.0")
    package, package_root = _temporary_package(tmp_path, monkeypatch)
    releases = package_root / "releases"
    first = _write_release(releases, registry_spec)
    second = _write_release(releases, version_two)
    first_name = f"releases/{first.name}"
    second_name = f"releases/{second.name}"

    single = ScenarioRegistry()
    assert single.load_package_resource(
        package,
        first_name,
        default=True,
    ) == registry_spec
    assert single.get("registry_fixture") == registry_spec

    batch = ScenarioRegistry()
    loaded = batch.load_package_resources(
        package,
        (second_name, first_name),
        defaults=(("registry_fixture", "2.0.0"),),
    )
    assert loaded == (version_two, registry_spec)
    assert batch.get("registry_fixture") == version_two


def test_package_resource_failures_and_batches_never_partially_register(
    registry_spec: ScenarioSpec,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package, package_root = _temporary_package(tmp_path, monkeypatch)
    good = _write_release(package_root / "releases", registry_spec)
    good_name = f"releases/{good.name}"
    bad = _write_release(package_root / "bad", registry_spec, b"\xff")
    bad_name = f"bad/{bad.name}"
    registry = ScenarioRegistry()

    for resource in (
        "missing.scenario.json",
        "../outside.scenario.json",
        "/absolute.scenario.json",
        "bad\\resource.scenario.json",
        "releases",
    ):
        _assert_unchanged_failure(
            registry,
            ScenarioLoadError,
            lambda resource=resource: registry.load_package_resource(
                package,
                resource,
            ),
        )

    _assert_unchanged_failure(
        registry,
        ScenarioLoadError,
        lambda: registry.load_package_resource(
            f"missing_package_{uuid.uuid4().hex}",
            good.name,
        ),
    )
    _assert_unchanged_failure(
        registry,
        ScenarioLoadError,
        lambda: registry.load_package_resources(
            package,
            (good_name, bad_name),
        ),
    )
    _assert_unchanged_failure(
        registry,
        DuplicateScenarioError,
        lambda: registry.load_package_resources(
            package,
            (good_name, good_name),
        ),
    )


def test_compile_registered_uses_exact_identity_is_deterministic_and_rng_safe(
    registry_spec: ScenarioSpec,
) -> None:
    version_two = _copy_spec(registry_spec, spec_version="2.0.0")
    registry = ScenarioRegistry()
    registry.register_many((registry_spec, version_two))
    random_state = random.getstate()

    compiled = compile_registered(
        registry,
        "registry_fixture",
        "1.0.0",
        "trial-7",
        73,
        IncentiveCondition.MINIMAL,
    )
    repeated = compile_registered(
        registry,
        "registry_fixture",
        "1.0.0",
        "trial-7",
        73,
        IncentiveCondition.MINIMAL,
    )
    direct = compile_scenario(
        registry_spec,
        "trial-7",
        73,
        IncentiveCondition.MINIMAL,
    )
    other_version = compile_registered(
        registry,
        "registry_fixture",
        "2.0.0",
        "trial-7",
        73,
        IncentiveCondition.MINIMAL,
    )

    assert compiled == repeated == direct
    assert compiled.spec_version == "1.0.0"
    assert compiled.spec_hash == registry_spec.spec_hash
    assert other_version.spec_version == "2.0.0"
    assert other_version.spec_hash == version_two.spec_hash
    assert other_version.instance_hash != compiled.instance_hash
    assert random.getstate() == random_state


def test_compile_registered_type_condition_and_identity_failures_do_not_mutate(
    registry_spec: ScenarioSpec,
) -> None:
    registry = ScenarioRegistry()
    registry.register(registry_spec)
    arguments = (
        registry,
        "registry_fixture",
        "1.0.0",
        "trial-1",
        1,
    )

    _assert_unchanged_failure(
        registry,
        ValueError,
        lambda: compile_registered(
            *arguments,
            IncentiveCondition.HIGH_INCENTIVE,
        ),
    )
    _assert_unchanged_failure(
        registry,
        TypeError,
        lambda: compile_registered(
            *arguments,
            "minimal",  # type: ignore[arg-type]
        ),
    )
    _assert_unchanged_failure(
        registry,
        UnknownScenarioVersionError,
        lambda: compile_registered(
            registry,
            "registry_fixture",
            "9.0.0",
            "trial-1",
            1,
            IncentiveCondition.MINIMAL,
        ),
    )
    _assert_unchanged_failure(
        registry,
        ValueError,
        lambda: compile_registered(
            registry,
            "registry_fixture",
            "1.0.0",
            "trial-1",
            True,
            IncentiveCondition.MINIMAL,
        ),
    )
    _assert_unchanged_failure(
        registry,
        TypeError,
        lambda: compile_registered(
            object(),  # type: ignore[arg-type]
            "registry_fixture",
            "1.0.0",
            "trial-1",
            1,
            IncentiveCondition.MINIMAL,
        ),
    )
