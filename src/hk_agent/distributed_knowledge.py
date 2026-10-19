import hashlib
import json
import math
import os
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

KNOWLEDGE_SCHEMA_VERSION = 1
MAX_DOCUMENT_BYTES = 2 * 1024 * 1024
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")
_CONTRIBUTION_PATH_RE = re.compile(r"^knowledge/contributions/[0-9a-f]{64}\.json$")
_WORKER_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_PUBLIC_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,127}$")
_VERSION_KEYS = frozenset({"python", "numpy", "stable_baselines3", "sb3_contrib", "torch"})
_REWARD_VERSIONS = frozenset({"visual_motion_v1", "visual_motion_player_knowledge_v1", "reach_scene_change_v1", "explore_v1", "explore_v2", "explore_v3", "explore_v4"})
_TASK_CONTRACT_KEYS = frozenset(
    {
        "reward_version",
        "task_contract_schema_version",
        "observation_version",
        "action_names",
        "raw_action_indices",
        "actions",
        "config",
        "information_boundary",
        "reward_only_signals",
        "legal_action_mask",
        "detector_inputs",
        "privileged_state",
        "route_memory",
        "diagnostic_training",
        "claim",
        "ppo_hyperparameters",
    }
)
_TASK_CONFIG_KEYS = frozenset(
    {
        "horizon_steps",
        "pixel_delta_threshold",
        "motion_target_fraction",
        "target_window",
        "required_target_hits",
        "visual_change_scale",
        "time_cost",
        "diagnostic_target_bonus",
        "enemy_kill_reward",
        "enemy_kill_reward_budget",
        "shade_recovery_reward",
        "damage_penalty",
        "geo_gain_reward",
        "geo_spend_reward",
        "geo_reward_budget",
        "death_penalty",
        "run_failed_penalty",
        "crop_top_fraction",
        "crop_bottom_fraction",
        "crop_left_fraction",
        "crop_right_fraction",
        "warmup_frames",
        "descriptor_rows",
        "descriptor_columns",
        "dark_luma_absolute",
        "dark_luma_ratio",
        "dark_pixel_fraction",
        "white_luma_absolute",
        "white_pixel_fraction",
        "minimum_entropy_bits",
        "baseline_stable_distance",
        "transition_dark_frames",
        "new_scene_distance",
        "stable_frame_distance",
        "stable_frames",
        "novelty_shaping_budget",
        "success_reward",
        "terminate_on_scene_change",
        "novelty_reward", "novelty_threshold", "novelty_budget", "archive_max",
        "lifelong_archive_max", "novelty_persistence_steps",
        "shade_hit_reward", "hp_restored_reward", "stuck_seconds", "evaluation_interval",
        "hp_restored_budget", "scene_reward_budget", "time_cost_budget", "shade_hit_budget", "shade_approach_budget",
        "scene_change_reward", "lease_base_ms", "ent_coef", "gamma", "gae_lambda",
        "n_epochs", "target_kl", "learning_rate", "clip_range",
    }
)
_BUTTONS = frozenset({"LEFT", "RIGHT", "UP", "DOWN", "JUMP", "ATTACK", "CAST", "DASH", "SUPER_DASH", "DREAM_NAIL", "PAUSE", "MENU_SUBMIT", "MENU_CANCEL"})
_DURATIONS = frozenset({"base", "tap", "hold", "focus", "charge"})
_ACTION_NAMES = frozenset(
    {
        "NOOP",
        "LEFT",
        "RIGHT",
        "JUMP_TAP",
        "JUMP_HOLD",
        "LEFT+JUMP_TAP",
        "LEFT+JUMP_HOLD",
        "RIGHT+JUMP_TAP",
        "RIGHT+JUMP_HOLD",
        "ATTACK",
        "LEFT+ATTACK",
        "RIGHT+ATTACK",
        "UP+ATTACK",
        "DOWN+ATTACK",
        "JUMP+ATTACK",
        "LEFT+JUMP+ATTACK",
        "RIGHT+JUMP+ATTACK",
        "CAST",
        "UP+CAST",
        "DOWN+CAST",
        "FOCUS_HOLD",
        "NAIL_ART_CHARGE",
        "LEFT+NAIL_ART_CHARGE",
        "RIGHT+NAIL_ART_CHARGE",
        "DASH", "LEFT+DASH", "RIGHT+DASH", "DOWN+DASH", "SUPER_DASH_HOLD", "DREAM_NAIL",
        "UP+NAIL_ART_CHARGE",
        "DOWN+NAIL_ART_CHARGE",
        "INTERACT", "MENU_UP", "MENU_DOWN", "MENU_LEFT", "MENU_RIGHT", "MENU_SUBMIT", "MENU_CANCEL", "OPEN_CHARMS",
    }
)
_INFORMATION_BOUNDARY = frozenset({"pixels", "applied_action_history"})
_REWARD_ONLY_SIGNALS = frozenset(
    {
        "enemy_kills_since_last_frame",
        "hp_lost_since_last_frame",
        "geo_gained_since_last_frame",
        "geo_spent_since_last_frame",
        "shade_recovered",
        "hp_restored_since_last_frame",
        "shade_hits_since_last_frame",
        "shade_distance",
        "scene_name",
        "enemy_type",
        "enemy_events_since_last_frame",
        "death",
        "run_failed",
    }
)
_DETECTOR_INPUTS = frozenset({"cropped_rgb_pixels", "temporal_pixel_descriptors"})
_CLAIMS = frozenset(
    {
        "motor-signal diagnostic; not game progress or task success",
        (
            "pixel-only stable visual-regime-change diagnostic; room transition "
            "not yet validated against a holdout corpus"
        ),
        "pixel-only exploration with bounded reward-only bridge signals",
    }
)
_ROUTE_LITERAL_FIELDS = {
    "node_identity": "nearest_visual_landmark_within_match_distance",
    "edges": "directed_observed_traversals",
    "revisits": "counted_and_allowed",
    "policy_integration": "external_planner_only",
    "reward_integration": "none",
    "edge_payload": "shortest_observed_temporal_action_trace",
    "confidence": "observed_traversal_count_capped_at_three",
    "single_writer": "optimistic_revision_check",
}
_SEMANTIC_FILE_HASHES = frozenset(
    {
        "game_executable",
        "mod_dll",
        "bootstrap_plan",
        "readiness_reference",
        "curriculum_source",
        "temporal_actions_source",
        "environment_source",
        "policy_runtime_source",
        "policy_artifact_source",
        "training_lock",
    }
)
_TRAINING_KEYS = {
    "algorithm",
    "seed",
    "reward_version",
    "status",
    "completed_timesteps",
    "episode_count",
    "gameplay_run_index",
    "gameplay_run_failures",
}


class KnowledgeValidationError(ValueError):
    """Raised when public knowledge does not satisfy the closed schema."""


def canonical_document_sha256(value: object) -> str:
    """Return the stable digest used for registry compare-and-swap operations."""

    return _sha256_json(value)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _expect_dict(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise KnowledgeValidationError(f"{label} must be a JSON object with string keys")
    return value


def _expect_exact_keys(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise KnowledgeValidationError(
            f"{label} keys must be exactly {sorted(keys)}; got {sorted(value)}"
        )


def _expect_allowed_keys(
    value: Mapping[str, Any], allowed: frozenset[str] | set[str], label: str
) -> None:
    unexpected = set(value) - set(allowed)
    if unexpected:
        raise KnowledgeValidationError(f"{label} contains unsupported keys: {sorted(unexpected)}")


def _public_identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or not _PUBLIC_ID_RE.fullmatch(value):
        raise KnowledgeValidationError(f"{label} must be a safe public identifier")
    return value


def _public_game_profile(value: object, *, from_manifest: bool) -> str:
    if from_manifest:
        profile = _public_identifier(value, "provenance.game_profile")
        return f"sha256:{hashlib.sha256(profile.encode('utf-8')).hexdigest()}"
    if (
        not isinstance(value, str)
        or not value.startswith("sha256:")
        or not _SHA256_RE.fullmatch(value.removeprefix("sha256:"))
    ):
        raise KnowledgeValidationError("public game_profile must be a SHA-256 label")
    return value


def _finite_number(value: object, label: str) -> int | float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise KnowledgeValidationError(f"{label} must be a finite number")
    return value


def _public_dimensions(value: object) -> dict[str, Any]:
    dimensions = _expect_dict(value, "observation_dimensions")
    _expect_allowed_keys(dimensions, {"capture", "policy"}, "observation_dimensions")
    result: dict[str, Any] = {}
    allowed = {
        "capture": {"height", "width", "frame_stack", "channels_per_frame"},
        "policy": {"height", "width", "channels", "action_history_length"},
    }
    for section_name, section_value in dimensions.items():
        if section_value is None and section_name == "policy":
            result[section_name] = None
            continue
        section = _expect_dict(section_value, f"observation_dimensions.{section_name}")
        _expect_allowed_keys(
            section, allowed[section_name], f"observation_dimensions.{section_name}"
        )
        clean: dict[str, int] = {}
        for name, raw in section.items():
            if not isinstance(raw, int) or isinstance(raw, bool) or raw < 0:
                raise KnowledgeValidationError(
                    f"observation_dimensions.{section_name}.{name} must be non-negative integer"
                )
            clean[name] = raw
        result[section_name] = clean
    return result


def _public_route_memory(value: object, *, from_manifest: bool) -> dict[str, Any]:
    route = _expect_dict(value, "task_contract.route_memory")
    allowed = {
        "enabled",
        "config",
    }
    _expect_allowed_keys(route, allowed, "task_contract.route_memory")
    if not isinstance(route.get("enabled"), bool):
        raise KnowledgeValidationError("task_contract.route_memory.enabled must be boolean")
    config = _expect_dict(route.get("config"), "task_contract.route_memory.config")
    config_keys = {
        "schema_version",
        "inputs",
        *_ROUTE_LITERAL_FIELDS,
        "privileged_state",
        "match_distance",
        "scope",
    }
    _expect_allowed_keys(config, config_keys, "task_contract.route_memory.config")
    result_config: dict[str, Any] = {}
    schema_version = config.get("schema_version")
    if schema_version != 1:
        raise KnowledgeValidationError("route memory schema_version must be 1")
    result_config["schema_version"] = schema_version
    inputs = config.get("inputs")
    if inputs != ["pixel_scene_descriptor", "confirmed_stable_visual_regime_change"]:
        raise KnowledgeValidationError("route memory inputs are not supported")
    result_config["inputs"] = list(inputs)
    for name, expected in _ROUTE_LITERAL_FIELDS.items():
        if config.get(name) != expected:
            raise KnowledgeValidationError(f"route memory {name} is not supported")
        result_config[name] = expected
    if config.get("privileged_state") != []:
        raise KnowledgeValidationError("route memory privileged_state must be empty")
    result_config["privileged_state"] = []
    if "match_distance" in config:
        result_config["match_distance"] = _finite_number(
            config["match_distance"], "route memory match_distance"
        )
    if "scope" in config:
        scope = _expect_dict(config["scope"], "route memory scope")
        allowed_scope = {
            "game_profile",
            "save_source",
            "save_slot",
            "detector_signature",
        }
        # campaign_id is deliberately private: current runners use an absolute output path.
        source_allowed = allowed_scope | ({"campaign_id"} if from_manifest else set())
        _expect_allowed_keys(scope, source_allowed, "route memory scope")
        clean_scope: dict[str, Any] = {}
        for name in sorted(allowed_scope & set(scope)):
            raw = scope[name]
            if name == "save_slot":
                if not isinstance(raw, int) or isinstance(raw, bool) or not 1 <= raw <= 4:
                    raise KnowledgeValidationError("route memory save_slot must be in [1, 4]")
                clean_scope[name] = raw
            elif name == "save_source":
                if raw not in {"new_game", "existing_save", "continuous_playthrough"}:
                    raise KnowledgeValidationError("route memory save_source is not supported")
                clean_scope[name] = raw
            elif name == "game_profile":
                clean_scope[name] = _public_game_profile(raw, from_manifest=from_manifest)
            else:
                if not isinstance(raw, str) or not _SHA256_RE.fullmatch(raw):
                    raise KnowledgeValidationError(
                        "route memory detector_signature must be a SHA-256"
                    )
                clean_scope[name] = raw
        result_config["scope"] = clean_scope
    return {"enabled": route["enabled"], "config": result_config}


def _public_task_contract(value: object, *, from_manifest: bool) -> dict[str, Any]:
    contract = _expect_dict(value, "task_contract")
    _expect_allowed_keys(contract, _TASK_CONTRACT_KEYS, "task_contract")
    required = {"reward_version", "action_names", "config"}
    if not required <= set(contract):
        raise KnowledgeValidationError("task_contract is missing required public fields")
    result: dict[str, Any] = {}
    if contract["reward_version"] not in _REWARD_VERSIONS:
        raise KnowledgeValidationError("task_contract.reward_version is not supported")
    if "legal_action_mask" in contract:
        mask = _expect_dict(contract["legal_action_mask"], "task_contract.legal_action_mask")
        _expect_allowed_keys(mask, {"version", "source"}, "task_contract.legal_action_mask")
        if mask not in (
            {"version": "action_availability_v1", "source": "frame_action_availability"},
            {"version": "action_availability_v2", "source": "frame_action_availability"},
            {"version": "action_availability_v3", "source": "frame_action_availability"},
        ):
            raise KnowledgeValidationError("task_contract.legal_action_mask is unsupported")
        result["legal_action_mask"] = dict(mask)
    result["reward_version"] = contract["reward_version"]
    schema_version = contract.get("task_contract_schema_version", 1)
    if isinstance(schema_version, bool) or not isinstance(schema_version, int) or schema_version not in {1, 2, 3, 4, 5, 6, 7, 8}:
        raise KnowledgeValidationError("task_contract_schema_version is unsupported")
    result["task_contract_schema_version"] = schema_version
    if "observation_version" in contract:
        observation_version = contract["observation_version"]
        if not isinstance(observation_version, str) or len(observation_version) > 80:
            raise KnowledgeValidationError("task_contract.observation_version is invalid")
        result["observation_version"] = observation_version
    action_names = contract["action_names"]
    if not isinstance(action_names, list) or not all(
        name in _ACTION_NAMES for name in action_names
    ):
        raise KnowledgeValidationError("task_contract.action_names contains unsupported actions")
    result["action_names"] = list(action_names)
    if "raw_action_indices" in contract:
        indices = contract["raw_action_indices"]
        if not isinstance(indices, list) or not all(
            isinstance(index, int) and not isinstance(index, bool) and 0 <= index <= 1023
            for index in indices
        ):
            raise KnowledgeValidationError("task_contract.raw_action_indices is invalid")
        result["raw_action_indices"] = list(indices)
    if "actions" in contract:
        actions = contract["actions"]
        if not isinstance(actions, list) or len(actions) > len(_ACTION_NAMES):
            raise KnowledgeValidationError("task_contract.actions must be a bounded list")
        clean_actions: list[dict[str, Any]] = []
        for index, raw in enumerate(actions):
            action = _expect_dict(raw, f"task_contract.actions[{index}]")
            _expect_exact_keys(
                action,
                {"name", "buttons", "raw_action_index", "duration", "lease_ms", "release_after"},
                f"task_contract.actions[{index}]",
            )
            if action["name"] not in _ACTION_NAMES:
                raise KnowledgeValidationError("task_contract action name is unsupported")
            buttons = action["buttons"]
            if not isinstance(buttons, list) or not all(button in _BUTTONS for button in buttons):
                raise KnowledgeValidationError("task_contract action buttons are unsupported")
            raw_index, lease_ms = action["raw_action_index"], action["lease_ms"]
            if (
                not isinstance(raw_index, int)
                or isinstance(raw_index, bool)
                or not 0 <= raw_index <= 1023
            ):
                raise KnowledgeValidationError("task_contract action index is invalid")
            if (
                not isinstance(lease_ms, int)
                or isinstance(lease_ms, bool)
                or not 1 <= lease_ms <= 2_000
            ):
                raise KnowledgeValidationError("task_contract action lease is invalid")
            if action["duration"] not in _DURATIONS or action["release_after"] is not True:
                raise KnowledgeValidationError("task_contract action timing is unsupported")
            clean_actions.append(dict(action))
        result["actions"] = clean_actions
    config = _expect_dict(contract["config"], "task_contract.config")
    _expect_allowed_keys(config, _TASK_CONFIG_KEYS, "task_contract.config")
    clean_config: dict[str, Any] = {}
    for name, raw in config.items():
        if raw is None and name == "horizon_steps":
            clean_config[name] = None
        elif isinstance(raw, bool):
            clean_config[name] = raw
        else:
            clean_config[name] = _finite_number(raw, f"task_contract.config.{name}")
    result["config"] = clean_config
    list_fields = {
        "information_boundary": _INFORMATION_BOUNDARY,
        "reward_only_signals": _REWARD_ONLY_SIGNALS,
        "detector_inputs": _DETECTOR_INPUTS,
    }
    for name, allowed_values in list_fields.items():
        if name not in contract:
            continue
        raw = contract[name]
        if not isinstance(raw, list) or not all(item in allowed_values for item in raw):
            raise KnowledgeValidationError(f"task_contract.{name} contains unsupported values")
        result[name] = list(raw)
    if "privileged_state" in contract:
        if contract["privileged_state"] != []:
            raise KnowledgeValidationError("task_contract.privileged_state must be empty")
        result["privileged_state"] = []
    if "route_memory" in contract:
        result["route_memory"] = _public_route_memory(
            contract["route_memory"], from_manifest=from_manifest
        )
    if "diagnostic_training" in contract:
        if not isinstance(contract["diagnostic_training"], bool):
            raise KnowledgeValidationError("task_contract.diagnostic_training must be boolean")
        result["diagnostic_training"] = contract["diagnostic_training"]
    if "claim" in contract:
        if contract["claim"] not in _CLAIMS:
            raise KnowledgeValidationError("task_contract.claim is not an approved public claim")
        result["claim"] = contract["claim"]
    if "ppo_hyperparameters" in contract:
        ppo = _expect_dict(contract["ppo_hyperparameters"], "task_contract.ppo_hyperparameters")
        allowed_ppo = {"ent_coef", "gamma", "gae_lambda", "n_epochs", "target_kl", "learning_rate", "clip_range"}
        _expect_allowed_keys(ppo, allowed_ppo, "task_contract.ppo_hyperparameters")
        result["ppo_hyperparameters"] = {
            key: _finite_number(value, f"task_contract.ppo_hyperparameters.{key}")
            for key, value in ppo.items()
        }
    return result


def _portable_origin(provenance: Mapping[str, Any], *, from_manifest: bool) -> dict[str, Any]:
    raw = provenance.get("campaign_origin", {})
    if not isinstance(raw, dict):
        return {}
    allowed_source_keys = {
        "schema_version",
        "save_source",
        "save_slot",
        "save_file_name",
        "base_plan_path",
        "resolved_plan_id",
        "resolved_plan_sha256",
        "resolved_steps",
        "subsequent_episode_bootstrap",
    }
    public_keys = allowed_source_keys - {"save_slot", "save_file_name", "base_plan_path"}
    _expect_allowed_keys(
        raw,
        allowed_source_keys if from_manifest else public_keys,
        "campaign_origin",
    )
    result: dict[str, Any] = {}
    if "schema_version" in raw:
        if raw["schema_version"] != 1:
            raise KnowledgeValidationError("campaign_origin.schema_version must be 1")
        result["schema_version"] = 1
    if "save_source" in raw:
        if raw["save_source"] not in {"new_game", "existing_save", "continuous_playthrough"}:
            raise KnowledgeValidationError("campaign_origin.save_source is not supported")
        result["save_source"] = raw["save_source"]
    # Plan identifiers and step labels are local, human-authored strings. Their
    # content is never public; the canonical plan hash carries the semantics.
    if "resolved_plan_sha256" in raw:
        digest = raw["resolved_plan_sha256"]
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            raise KnowledgeValidationError("campaign_origin.resolved_plan_sha256 is invalid")
        result["resolved_plan_sha256"] = digest
    steps = raw.get("resolved_steps")
    if steps is not None:
        if not isinstance(steps, list) or len(steps) > 64:
            raise KnowledgeValidationError("campaign_origin.resolved_steps must be a bounded list")
        clean_steps: list[dict[str, Any]] = []
        for index, step_value in enumerate(steps):
            step = _expect_dict(step_value, f"campaign_origin.resolved_steps[{index}]")
            allowed_step_keys = {"button", "lease_ms", "settle_seconds"}
            _expect_allowed_keys(
                step,
                allowed_step_keys | ({"label"} if from_manifest else set()),
                f"campaign_origin.resolved_steps[{index}]",
            )
            if "button" not in step:
                raise KnowledgeValidationError("campaign origin step is missing its button")
            button = step["button"]
            if not isinstance(button, str) or button not in _BUTTONS | {"SUBMIT", "CANCEL"}:
                raise KnowledgeValidationError("campaign origin step button is unsupported")
            clean: dict[str, Any] = {"button": button}
            if "lease_ms" in step:
                lease_ms = step["lease_ms"]
                if (
                    not isinstance(lease_ms, int)
                    or isinstance(lease_ms, bool)
                    or not 1 <= lease_ms <= 2_000
                ):
                    raise KnowledgeValidationError("campaign origin step lease is invalid")
                clean["lease_ms"] = lease_ms
            if "settle_seconds" in step:
                settle = _finite_number(step["settle_seconds"], "campaign origin settle_seconds")
                if not 0 <= settle <= 120:
                    raise KnowledgeValidationError(
                        "campaign origin settle_seconds is outside safe range"
                    )
                clean["settle_seconds"] = settle
            clean_steps.append(clean)
        result["resolved_steps"] = clean_steps
    subsequent = raw.get("subsequent_episode_bootstrap")
    if subsequent is not None:
        if not isinstance(subsequent, dict):
            raise KnowledgeValidationError("subsequent campaign bootstrap must be an object")
        result["subsequent_episode_bootstrap"] = _portable_origin(
            {"campaign_origin": subsequent}, from_manifest=from_manifest
        )
    return result


def _portable_source_hashes(provenance: Mapping[str, Any]) -> dict[str, str]:
    files = provenance.get("files", {})
    source_hashes: dict[str, str] = {}
    if not isinstance(files, dict):
        return source_hashes
    for name, record in sorted(files.items()):
        if name not in _SEMANTIC_FILE_HASHES or not isinstance(record, dict):
            continue
        digest = record.get("sha256")
        if isinstance(digest, str) and _SHA256_RE.fullmatch(digest):
            source_hashes[name] = digest
    return source_hashes


def _fingerprint_from_portable_parts(
    task_contract: Mapping[str, Any],
    *,
    game_profile: str | None,
    campaign_origin: Mapping[str, Any],
    source_hashes: Mapping[str, str],
    observation_dimensions: Mapping[str, Any],
) -> str:
    return _sha256_json(
        {
            "task_contract": dict(task_contract),
            "game_profile": game_profile,
            "campaign_origin": dict(campaign_origin),
            "source_hashes": dict(source_hashes),
            "observation_dimensions": dict(observation_dimensions),
        }
    )


def portable_compatibility_fingerprint(
    task_contract: Mapping[str, Any], provenance: Mapping[str, Any]
) -> str:
    """Hash cross-machine semantics without publishing the supplied contract.

    Training tasks may use private or experimental contracts that are not eligible for
    the public registry. They still need a stable local resume fingerprint. Publication
    applies the closed schema separately in :func:`contribution_from_manifest`.
    """

    dimensions = _public_dimensions(provenance.get("observation_dimensions", {}))
    game_profile = provenance.get("game_profile")
    clean_profile = (
        _public_game_profile(game_profile, from_manifest=True) if game_profile is not None else None
    )
    try:
        fingerprint_contract: Mapping[str, Any] = _public_task_contract(
            task_contract, from_manifest=True
        )
    except KnowledgeValidationError:
        fingerprint_contract = task_contract
    return _fingerprint_from_portable_parts(
        fingerprint_contract,
        game_profile=clean_profile,
        campaign_origin=_portable_origin(provenance, from_manifest=True),
        source_hashes=_portable_source_hashes(provenance),
        observation_dimensions=dimensions,
    )


def contribution_from_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Create a public, allowlisted summary from a private training manifest."""

    task_contract = _public_task_contract(manifest.get("task_contract"), from_manifest=True)
    provenance = _expect_dict(manifest.get("provenance"), "manifest.provenance")
    checkpoints: list[dict[str, Any]] = []
    for checkpoint in manifest.get("checkpoints", []):
        if not isinstance(checkpoint, dict):
            continue
        step, digest = checkpoint.get("step"), checkpoint.get("sha256")
        if (
            isinstance(step, int)
            and not isinstance(step, bool)
            and isinstance(digest, str)
            and _SHA256_RE.fullmatch(digest)
        ):
            checkpoints.append({"step": step, "sha256": digest})

    versions = _expect_dict(manifest.get("versions", {}), "manifest.versions")
    _expect_allowed_keys(versions, _VERSION_KEYS, "manifest.versions")
    # Version values are intentionally not published. They are arbitrary strings
    # supplied by the local runtime and are not part of compatibility semantics.
    clean_versions: dict[str, str] = {}
    game_profile = provenance.get("game_profile")
    clean_profile = (
        _public_game_profile(game_profile, from_manifest=True) if game_profile is not None else None
    )
    dimensions = _public_dimensions(provenance.get("observation_dimensions", {}))
    campaign_origin = _portable_origin(provenance, from_manifest=True)
    source_hashes = _portable_source_hashes(provenance)
    body: dict[str, Any] = {
        "schema_version": KNOWLEDGE_SCHEMA_VERSION,
        "compatibility_fingerprint": _fingerprint_from_portable_parts(
            task_contract,
            game_profile=clean_profile,
            campaign_origin=campaign_origin,
            source_hashes=source_hashes,
            observation_dimensions=dimensions,
        ),
        "training": {
            "algorithm": manifest.get("algorithm"),
            "seed": manifest.get("seed"),
            "reward_version": manifest.get("reward_version"),
            "status": manifest.get("status"),
            "completed_timesteps": manifest.get("completed_timesteps", 0),
            "episode_count": manifest.get("episode_count", 0),
            "gameplay_run_index": manifest.get("gameplay_run_index", 0),
            "gameplay_run_failures": manifest.get("gameplay_run_failures", 0),
        },
        "task_contract": task_contract,
        "portable_provenance": {
            "game_profile": clean_profile,
            "campaign_origin": campaign_origin,
            "observation_dimensions": dimensions,
            "source_hashes": source_hashes,
        },
        "versions": clean_versions,
        "checkpoint_fingerprints": sorted(checkpoints, key=lambda item: item["step"]),
    }
    contribution_id = _sha256_json(body)
    contribution = {"contribution_id": contribution_id, **body}
    validate_contribution(contribution)
    return contribution


def validate_contribution(value: object) -> dict[str, Any]:
    contribution = _expect_dict(value, "contribution")
    _expect_exact_keys(
        contribution,
        {
            "contribution_id",
            "schema_version",
            "compatibility_fingerprint",
            "training",
            "task_contract",
            "portable_provenance",
            "versions",
            "checkpoint_fingerprints",
        },
        "contribution",
    )
    if contribution["schema_version"] != KNOWLEDGE_SCHEMA_VERSION:
        raise KnowledgeValidationError("unsupported contribution schema_version")
    for key in ("contribution_id", "compatibility_fingerprint"):
        if not isinstance(contribution[key], str) or not _SHA256_RE.fullmatch(contribution[key]):
            raise KnowledgeValidationError(f"contribution.{key} must be lowercase SHA-256")
    training = _expect_dict(contribution["training"], "contribution.training")
    _expect_exact_keys(training, _TRAINING_KEYS, "contribution.training")
    if training["algorithm"] not in {"PPO", "MaskablePPO"}:
        raise KnowledgeValidationError("training.algorithm is not supported")
    if training["reward_version"] not in _REWARD_VERSIONS:
        raise KnowledgeValidationError("training.reward_version is not supported")
    if training["status"] not in {"RUNNING", "COMPLETED", "CANCELLED", "FAILED"}:
        raise KnowledgeValidationError("training.status is not supported")
    for key in (
        "seed",
        "completed_timesteps",
        "episode_count",
        "gameplay_run_index",
        "gameplay_run_failures",
    ):
        if not isinstance(training[key], int) or isinstance(training[key], bool):
            raise KnowledgeValidationError(f"training.{key} must be an integer")
    task_contract = _expect_dict(contribution["task_contract"], "contribution.task_contract")
    if _public_task_contract(task_contract, from_manifest=False) != task_contract:
        raise KnowledgeValidationError("task_contract is not in canonical public form")
    if training["reward_version"] != task_contract["reward_version"]:
        raise KnowledgeValidationError("training and task reward versions differ")
    portable = _expect_dict(contribution["portable_provenance"], "portable_provenance")
    _expect_exact_keys(
        portable,
        {"game_profile", "campaign_origin", "observation_dimensions", "source_hashes"},
        "portable_provenance",
    )
    game_profile = portable["game_profile"]
    if game_profile is not None:
        _public_game_profile(game_profile, from_manifest=False)
    origin = _expect_dict(portable["campaign_origin"], "portable_provenance.campaign_origin")
    if _portable_origin({"campaign_origin": origin}, from_manifest=False) != origin:
        raise KnowledgeValidationError("campaign_origin is not in canonical public form")
    dimensions = _expect_dict(
        portable["observation_dimensions"], "portable_provenance.observation_dimensions"
    )
    if _public_dimensions(dimensions) != dimensions:
        raise KnowledgeValidationError("observation_dimensions is not in canonical public form")
    source_hashes = _expect_dict(portable["source_hashes"], "portable_provenance.source_hashes")
    for name, digest in source_hashes.items():
        if name not in _SEMANTIC_FILE_HASHES:
            raise KnowledgeValidationError("portable_provenance contains a non-semantic file hash")
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            raise KnowledgeValidationError("portable source hash is invalid")
    versions = _expect_dict(contribution["versions"], "contribution.versions")
    if versions:
        raise KnowledgeValidationError("public contribution versions must be empty")
    expected_fingerprint = _fingerprint_from_portable_parts(
        task_contract,
        game_profile=game_profile,
        campaign_origin=origin,
        source_hashes=source_hashes,
        observation_dimensions=dimensions,
    )
    if contribution["compatibility_fingerprint"] != expected_fingerprint:
        raise KnowledgeValidationError(
            "compatibility_fingerprint does not match portable semantics"
        )
    fingerprints = contribution["checkpoint_fingerprints"]
    if not isinstance(fingerprints, list):
        raise KnowledgeValidationError("checkpoint_fingerprints must be a list")
    for item in fingerprints:
        record = _expect_dict(item, "checkpoint fingerprint")
        _expect_exact_keys(record, {"step", "sha256"}, "checkpoint fingerprint")
        if (
            not isinstance(record["step"], int)
            or isinstance(record["step"], bool)
            or record["step"] < 0
        ):
            raise KnowledgeValidationError("checkpoint step must be a non-negative integer")
        if not isinstance(record["sha256"], str) or not _SHA256_RE.fullmatch(record["sha256"]):
            raise KnowledgeValidationError("checkpoint sha256 is invalid")
    body = dict(contribution)
    claimed_id = body.pop("contribution_id")
    if _sha256_json(body) != claimed_id:
        raise KnowledgeValidationError("contribution_id does not match canonical content")
    forbidden = re.compile(r'(?i)([a-z]:\\|/users/|\\users\\|\.png|\.dat(?:\b|\.)|"path"\s*:)')
    if forbidden.search(_canonical_bytes(contribution).decode("utf-8")):
        raise KnowledgeValidationError(
            "contribution contains a local path, save, or screen reference"
        )
    return contribution


def validate_index(value: object) -> dict[str, Any]:
    index = _expect_dict(value, "knowledge index")
    _expect_exact_keys(index, {"schema_version", "contributions"}, "knowledge index")
    if index["schema_version"] != KNOWLEDGE_SCHEMA_VERSION:
        raise KnowledgeValidationError("unsupported knowledge index schema_version")
    entries = index["contributions"]
    if not isinstance(entries, list):
        raise KnowledgeValidationError("knowledge index contributions must be a list")
    if len(entries) > 1_000:
        raise KnowledgeValidationError("knowledge index exceeds 1000 contributions")
    seen: set[str] = set()
    for raw in entries:
        entry = _expect_dict(raw, "knowledge index entry")
        _expect_exact_keys(
            entry,
            {"contribution_id", "path", "sha256", "compatibility_fingerprint"},
            "knowledge index entry",
        )
        contribution_id = entry["contribution_id"]
        if not isinstance(contribution_id, str) or not _SHA256_RE.fullmatch(contribution_id):
            raise KnowledgeValidationError("index contribution_id is invalid")
        expected_path = f"knowledge/contributions/{contribution_id}.json"
        if entry["path"] != expected_path or not _CONTRIBUTION_PATH_RE.fullmatch(entry["path"]):
            raise KnowledgeValidationError("index contribution path is not canonical")
        for key in ("sha256", "compatibility_fingerprint"):
            if not isinstance(entry[key], str) or not _SHA256_RE.fullmatch(entry[key]):
                raise KnowledgeValidationError(f"index {key} is invalid")
        if contribution_id in seen:
            raise KnowledgeValidationError("knowledge index contains duplicate contributions")
        seen.add(contribution_id)
    return index


def _validate_artifact(value: object, label: str = "artifact") -> dict[str, Any]:
    artifact = _expect_dict(value, label)
    _expect_exact_keys(artifact, {"uri", "sha256", "size_bytes", "format"}, label)
    uri = artifact["uri"]
    if not isinstance(uri, str) or len(uri) > 2_048:
        raise KnowledgeValidationError(f"{label}.uri must be an HTTPS URL")
    parsed = urlsplit(uri)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise KnowledgeValidationError(
            f"{label}.uri must be HTTPS without credentials, query, or fragment"
        )
    digest = artifact["sha256"]
    if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
        raise KnowledgeValidationError(f"{label}.sha256 must be lowercase SHA-256")
    size = artifact["size_bytes"]
    if not isinstance(size, int) or isinstance(size, bool) or not 0 < size <= 2 * 1024**3:
        raise KnowledgeValidationError(f"{label}.size_bytes must be between 1 and 2 GiB")
    if artifact["format"] not in {"sb3-zip", "safetensors"}:
        raise KnowledgeValidationError(f"{label}.format is not supported")
    return artifact


def _validate_parent_champion(value: object) -> dict[str, Any]:
    parent = _expect_dict(value, "candidate.parent_champion")
    _expect_exact_keys(parent, {"generation", "sha256"}, "candidate.parent_champion")
    generation = parent["generation"]
    if not isinstance(generation, int) or isinstance(generation, bool) or generation < 0:
        raise KnowledgeValidationError("candidate parent generation must be non-negative")
    digest = parent["sha256"]
    if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
        raise KnowledgeValidationError("candidate parent sha256 must be lowercase SHA-256")
    return parent


def candidate_from_contribution(
    contribution: Mapping[str, Any],
    *,
    worker_id: str,
    artifact_uri: str,
    artifact_sha256: str,
    artifact_size_bytes: int,
    artifact_format: str,
    trust_tier: str,
    parent_generation: int,
    parent_sha256: str,
) -> dict[str, Any]:
    """Build a content-addressed candidate without reading its model artifact."""

    clean_contribution = validate_contribution(dict(contribution))
    body: dict[str, Any] = {
        "schema_version": KNOWLEDGE_SCHEMA_VERSION,
        "worker_id": worker_id,
        "trust_tier": trust_tier,
        "contribution_id": clean_contribution["contribution_id"],
        "compatibility_fingerprint": clean_contribution["compatibility_fingerprint"],
        "parent_champion": {"generation": parent_generation, "sha256": parent_sha256},
        "artifact": {
            "uri": artifact_uri,
            "sha256": artifact_sha256,
            "size_bytes": artifact_size_bytes,
            "format": artifact_format,
        },
    }
    candidate = {"candidate_id": _sha256_json(body), **body}
    return validate_candidate(candidate)


def validate_candidate(value: object) -> dict[str, Any]:
    candidate = _expect_dict(value, "candidate")
    _expect_exact_keys(
        candidate,
        {
            "candidate_id",
            "schema_version",
            "worker_id",
            "trust_tier",
            "contribution_id",
            "compatibility_fingerprint",
            "parent_champion",
            "artifact",
        },
        "candidate",
    )
    if candidate["schema_version"] != KNOWLEDGE_SCHEMA_VERSION:
        raise KnowledgeValidationError("unsupported candidate schema_version")
    worker_id = candidate["worker_id"]
    if not isinstance(worker_id, str) or not _WORKER_ID_RE.fullmatch(worker_id):
        raise KnowledgeValidationError("candidate.worker_id is not a safe public identifier")
    if candidate["trust_tier"] not in {"owned", "quarantined", "external-untrusted"}:
        raise KnowledgeValidationError("candidate.trust_tier is not supported")
    for key in ("candidate_id", "contribution_id", "compatibility_fingerprint"):
        if not isinstance(candidate[key], str) or not _SHA256_RE.fullmatch(candidate[key]):
            raise KnowledgeValidationError(f"candidate.{key} must be lowercase SHA-256")
    _validate_parent_champion(candidate["parent_champion"])
    _validate_artifact(candidate["artifact"], "candidate.artifact")
    body = dict(candidate)
    claimed_id = body.pop("candidate_id")
    if _sha256_json(body) != claimed_id:
        raise KnowledgeValidationError("candidate_id does not match canonical content")
    return candidate


def initial_champion() -> dict[str, Any]:
    return {
        "schema_version": KNOWLEDGE_SCHEMA_VERSION,
        "generation": 0,
        "candidate_id": None,
        "contribution_id": None,
        "compatibility_fingerprint": None,
        "artifact": None,
        "parent_champion_sha256": None,
        "evaluation": None,
    }


def validate_champion(value: object) -> dict[str, Any]:
    champion = _expect_dict(value, "champion")
    keys = {
        "schema_version",
        "generation",
        "candidate_id",
        "contribution_id",
        "compatibility_fingerprint",
        "artifact",
        "parent_champion_sha256",
        "evaluation",
    }
    _expect_exact_keys(champion, keys, "champion")
    if champion["schema_version"] != KNOWLEDGE_SCHEMA_VERSION:
        raise KnowledgeValidationError("unsupported champion schema_version")
    generation = champion["generation"]
    if not isinstance(generation, int) or isinstance(generation, bool) or generation < 0:
        raise KnowledgeValidationError("champion.generation must be non-negative")
    nullable_hashes = (
        "candidate_id",
        "contribution_id",
        "compatibility_fingerprint",
        "parent_champion_sha256",
    )
    for key in nullable_hashes:
        item = champion[key]
        if item is not None and (not isinstance(item, str) or not _SHA256_RE.fullmatch(item)):
            raise KnowledgeValidationError(f"champion.{key} must be null or lowercase SHA-256")
    if generation == 0:
        if champion != initial_champion():
            raise KnowledgeValidationError("generation-zero champion must be canonical and empty")
    else:
        if any(champion[key] is None for key in nullable_hashes) or champion["artifact"] is None:
            raise KnowledgeValidationError("promoted champion fields cannot be null")
        artifact = _validate_artifact(champion["artifact"], "champion.artifact")
        if artifact["format"] != "safetensors":
            raise KnowledgeValidationError(
                "promoted champion artifact must use the non-executable safetensors format"
            )
        evaluation = _expect_dict(champion["evaluation"], "champion.evaluation")
        _expect_exact_keys(
            evaluation,
            {
                "report_sha256",
                "suite_sha256",
                "policy_sha256",
                "candidate_id",
                "parent_champion_sha256",
                "evaluator_commit",
            },
            "champion.evaluation",
        )
        for key in (
            "report_sha256",
            "suite_sha256",
            "policy_sha256",
            "candidate_id",
            "parent_champion_sha256",
        ):
            if not isinstance(evaluation[key], str) or not _SHA256_RE.fullmatch(evaluation[key]):
                raise KnowledgeValidationError(f"champion evaluation {key} is invalid")
        commit = evaluation["evaluator_commit"]
        if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise KnowledgeValidationError("champion evaluator_commit must be a full Git SHA-1")
        if evaluation["policy_sha256"] != champion["artifact"]["sha256"]:
            raise KnowledgeValidationError(
                "champion evaluation policy hash does not match artifact"
            )
        if evaluation["candidate_id"] != champion["candidate_id"]:
            raise KnowledgeValidationError("champion evaluation candidate binding differs")
        if evaluation["parent_champion_sha256"] != champion["parent_champion_sha256"]:
            raise KnowledgeValidationError("champion evaluation parent binding differs")
    return champion


def promote_candidate(
    champion: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    expected_generation: int,
    expected_champion_sha256: str,
    evaluation_report_sha256: str,
    evaluation_suite_sha256: str,
    evaluator_commit: str,
) -> dict[str, Any]:
    """Reject promotion until evaluator attestations have a verifiable schema.

    Candidate trust tiers are supplied by the producer and therefore cannot authorize a
    state transition. The arguments remain accepted so older callers fail closed with a
    stable error instead of silently bypassing the missing authority boundary.
    """

    validate_champion(dict(champion))
    validate_candidate(dict(candidate))
    del (
        expected_generation,
        expected_champion_sha256,
        evaluation_report_sha256,
        evaluation_suite_sha256,
        evaluator_commit,
    )
    raise KnowledgeValidationError(
        "champion promotion is disabled until evaluator attestation is implemented"
    )


def _load_json(path: Path) -> object:
    if not path.is_file() or path.is_symlink():
        raise KnowledgeValidationError(f"expected a regular JSON file: {path}")
    if path.stat().st_size > MAX_DOCUMENT_BYTES:
        raise KnowledgeValidationError(f"JSON document exceeds size limit: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise KnowledgeValidationError(f"cannot read JSON document {path}: {error}") from error


def validate_knowledge_tree(root: Path) -> dict[str, Any]:
    resolved = root.resolve(strict=True)
    index = validate_index(_load_json(resolved / "index.json"))
    for entry in index["contributions"]:
        relative = PurePosixPath(entry["path"])
        path = resolved / Path(*relative.parts[1:])
        contribution = validate_contribution(_load_json(path))
        if contribution["contribution_id"] != entry["contribution_id"]:
            raise KnowledgeValidationError("index and contribution IDs differ")
        if contribution["compatibility_fingerprint"] != entry["compatibility_fingerprint"]:
            raise KnowledgeValidationError("index and contribution fingerprints differ")
        if hashlib.sha256(_canonical_bytes(contribution)).hexdigest() != entry["sha256"]:
            raise KnowledgeValidationError("indexed contribution content hash differs")
    champion_path = resolved / "champion.json"
    if champion_path.exists():
        validate_champion(_load_json(champion_path))
    return index


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_canonical_bytes(value) + b"\n")
    temporary.replace(path)


def export_contribution(manifest_path: Path, knowledge_root: Path) -> dict[str, Any]:
    manifest = _expect_dict(_load_json(manifest_path), "training manifest")
    contribution = contribution_from_manifest(manifest)
    root = knowledge_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    index_path = root / "index.json"
    index: dict[str, Any]
    if index_path.exists():
        index = validate_index(_load_json(index_path))
    else:
        index = {"schema_version": KNOWLEDGE_SCHEMA_VERSION, "contributions": []}
    contribution_id = contribution["contribution_id"]
    path = root / "contributions" / f"{contribution_id}.json"
    _atomic_json(path, contribution)
    entry = {
        "contribution_id": contribution_id,
        "path": f"knowledge/contributions/{contribution_id}.json",
        "sha256": hashlib.sha256(_canonical_bytes(contribution)).hexdigest(),
        "compatibility_fingerprint": contribution["compatibility_fingerprint"],
    }
    retained = [
        item for item in index["contributions"] if item["contribution_id"] != contribution_id
    ]
    index["contributions"] = sorted(retained + [entry], key=lambda item: item["contribution_id"])
    _atomic_json(index_path, index)
    validate_knowledge_tree(root)
    return entry


def export_candidate(
    contribution_path: Path,
    output_root: Path,
    *,
    worker_id: str,
    artifact_uri: str,
    artifact_sha256: str,
    artifact_size_bytes: int,
    artifact_format: str,
    trust_tier: str,
    parent_generation: int,
    parent_sha256: str,
) -> dict[str, Any]:
    contribution = _expect_dict(_load_json(contribution_path), "contribution")
    candidate = candidate_from_contribution(
        contribution,
        worker_id=worker_id,
        artifact_uri=artifact_uri,
        artifact_sha256=artifact_sha256,
        artifact_size_bytes=artifact_size_bytes,
        artifact_format=artifact_format,
        trust_tier=trust_tier,
        parent_generation=parent_generation,
        parent_sha256=parent_sha256,
    )
    output = output_root.resolve() / "outbox" / "candidates" / f"{candidate['candidate_id']}.json"
    _atomic_json(output, candidate)
    return {"candidate_id": candidate["candidate_id"], "path": str(output)}


def validate_champion_file(path: Path) -> dict[str, Any]:
    champion = validate_champion(_load_json(path))
    return {
        "generation": champion["generation"],
        "sha256": canonical_document_sha256(champion),
        "candidate_id": champion["candidate_id"],
    }


def promote_candidate_file(
    champion_path: Path,
    candidate_path: Path,
    *,
    expected_generation: int,
    expected_champion_sha256: str,
    evaluation_report_sha256: str,
    evaluation_suite_sha256: str,
    evaluator_commit: str,
) -> dict[str, Any]:
    current = _expect_dict(_load_json(champion_path), "champion")
    candidate = _expect_dict(_load_json(candidate_path), "candidate")
    promoted = promote_candidate(
        current,
        candidate,
        expected_generation=expected_generation,
        expected_champion_sha256=expected_champion_sha256,
        evaluation_report_sha256=evaluation_report_sha256,
        evaluation_suite_sha256=evaluation_suite_sha256,
        evaluator_commit=evaluator_commit,
    )
    _atomic_json(champion_path, promoted)
    return {
        "generation": promoted["generation"],
        "sha256": canonical_document_sha256(promoted),
        "candidate_id": promoted["candidate_id"],
    }


def _validate_git_ref(ref: str) -> None:
    if not _REF_RE.fullmatch(ref) or ".." in ref or "//" in ref or ref.endswith("/") or "@{" in ref:
        raise KnowledgeValidationError("unsafe Git ref")


def _git_show(repo: Path, ref: str, path: str) -> bytes:
    if path not in {"knowledge/index.json", "knowledge/champion.json"} and not (
        _CONTRIBUTION_PATH_RE.fullmatch(path)
    ):
        raise KnowledgeValidationError("unsafe knowledge path")
    try:
        result = subprocess.run(
            [_git_executable(), "-C", str(repo.resolve()), "show", f"{ref}:{path}"],
            check=False,
            capture_output=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired as error:
        raise KnowledgeValidationError(f"git show timed out for {path}") from error
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise KnowledgeValidationError(f"git show failed for {path}: {message}")
    if len(result.stdout) > MAX_DOCUMENT_BYTES:
        raise KnowledgeValidationError(f"Git knowledge document exceeds size limit: {path}")
    return result.stdout


def _git_executable() -> str:
    override = os.environ.get("VOIDGYM_GIT_EXE")
    if override:
        candidate = Path(override).resolve(strict=False)
        if not candidate.is_file():
            raise KnowledgeValidationError("VOIDGYM_GIT_EXE is not a regular file")
        return str(candidate)
    if os.name == "nt":
        program_files = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
        git_for_windows = program_files / "Git" / "cmd" / "git.exe"
        if git_for_windows.is_file():
            return str(git_for_windows)
    return "git"


def sync_from_git_ref(repo: Path, ref: str, knowledge_root: Path) -> dict[str, Any]:
    """Import a validated branch snapshot without checkout, merge, hooks, or execution."""

    _validate_git_ref(ref)
    try:
        remote_index_raw = _git_show(repo, ref, "knowledge/index.json")
        remote_index = validate_index(json.loads(remote_index_raw.decode("utf-8")))
        remote_champion_raw = _git_show(repo, ref, "knowledge/champion.json")
        remote_champion = validate_champion(json.loads(remote_champion_raw.decode("utf-8")))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise KnowledgeValidationError("remote knowledge index is invalid JSON") from error

    staged: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for entry in remote_index["contributions"]:
        raw = _git_show(repo, ref, entry["path"])
        try:
            contribution = validate_contribution(json.loads(raw.decode("utf-8")))
        except (UnicodeError, json.JSONDecodeError) as error:
            raise KnowledgeValidationError("remote contribution is invalid JSON") from error
        canonical_hash = hashlib.sha256(_canonical_bytes(contribution)).hexdigest()
        if canonical_hash != entry["sha256"]:
            raise KnowledgeValidationError("remote contribution does not match its index hash")
        if contribution["contribution_id"] != entry["contribution_id"]:
            raise KnowledgeValidationError("remote contribution does not match its index ID")
        if contribution["compatibility_fingerprint"] != entry["compatibility_fingerprint"]:
            raise KnowledgeValidationError("remote contribution has a mismatched fingerprint")
        staged.append((entry, contribution))

    root = knowledge_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    local_index_path = root / "index.json"
    local_champion_path = root / "champion.json"
    if local_champion_path.exists():
        local_champion = validate_champion(_load_json(local_champion_path))
        local_generation = local_champion["generation"]
        remote_generation = remote_champion["generation"]
        if remote_generation < local_generation:
            raise KnowledgeValidationError("remote champion generation would roll back local state")
        if remote_generation == local_generation and remote_champion != local_champion:
            raise KnowledgeValidationError("champion fork detected at the same generation")
        if remote_generation == local_generation + 1 and (
            remote_champion["parent_champion_sha256"] != canonical_document_sha256(local_champion)
        ):
            raise KnowledgeValidationError("remote champion does not descend from local champion")
        if remote_generation > local_generation + 1:
            raise KnowledgeValidationError(
                "remote champion skipped generations; verify the promotion chain explicitly"
            )
    local_index: dict[str, Any]
    if local_index_path.exists():
        local_index = validate_index(_load_json(local_index_path))
    else:
        local_index = {"schema_version": KNOWLEDGE_SCHEMA_VERSION, "contributions": []}
    entries = {item["contribution_id"]: item for item in local_index["contributions"]}
    for entry, contribution in staged:
        existing = entries.get(entry["contribution_id"])
        if existing is not None and existing != entry:
            raise KnowledgeValidationError("immutable contribution ID has conflicting content")
        _atomic_json(root / "contributions" / f"{entry['contribution_id']}.json", contribution)
        entries[entry["contribution_id"]] = entry
    merged = {
        "schema_version": KNOWLEDGE_SCHEMA_VERSION,
        "contributions": sorted(entries.values(), key=lambda item: item["contribution_id"]),
    }
    _atomic_json(local_champion_path, remote_champion)
    _atomic_json(local_index_path, merged)
    validate_knowledge_tree(root)
    return {
        "imported": len(staged),
        "total": len(entries),
        "ref": ref,
        "champion_generation": remote_champion["generation"],
        "champion_sha256": canonical_document_sha256(remote_champion),
    }
