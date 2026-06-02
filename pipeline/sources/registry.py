from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


KNOWN_BUCKETS = {
    "code",
    "docs",
    "math",
    "english_web",
    "chinese_web",
    "agent_like",
    "markup_config",
    "repo_activity",
    "notebook",
    "unknown",
}

KNOWN_STATUSES = {"usable", "metadata_only", "probe_only", "disabled"}
KNOWN_RAW_FORMATS = {"parquet", "jsonl", "tree"}
KNOWN_LANGUAGE_MODES = {"fixed", "field", "path_parent"}
KNOWN_SAMPLING_STRATEGIES = {"per_source", "per_language"}
KNOWN_SPLIT_STRATEGIES = {"deterministic_shuffle"}


class SourceRegistryError(RuntimeError):
    pass


@dataclass(frozen=True)
class RawConfig:
    root: str
    format: str
    path_globs: list[str]
    manifest: str | None


@dataclass(frozen=True)
class ReadConfig:
    text_field: str | None
    id_field: str | None
    metadata_fields: list[str]


@dataclass(frozen=True)
class LanguageConfig:
    mode: str
    value: str | None
    field: str | None
    known_values: list[str]


@dataclass(frozen=True)
class CleaningConfig:
    profile: str


@dataclass(frozen=True)
class SamplingConfig:
    strategy: str
    default_train_docs_per_group: int
    default_val_docs_per_group: int
    split_strategy: str


@dataclass(frozen=True)
class SourceSpec:
    schema_version: str
    source_id: str
    display_name: str
    status: str
    raw: RawConfig
    read: ReadConfig
    default_bucket: str
    bucket_mapping: dict[str, Any]
    language: LanguageConfig
    cleaning: CleaningConfig
    sampling: SamplingConfig
    known_issues: list[str]


def _reject_extra(path: str, data: dict[str, Any], allowed: set[str]) -> None:
    extra = sorted(set(data) - allowed)
    if extra:
        raise SourceRegistryError(f"{path}: unexpected fields: {extra}")


def _mapping(path: str, data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise SourceRegistryError(f"{path}: expected a mapping")
    return data


def _required_mapping(path: str, data: dict[str, Any], key: str) -> dict[str, Any]:
    if key not in data:
        raise SourceRegistryError(f"{path}: missing required field {key!r}")
    return _mapping(f"{path}.{key}", data[key])


def _required_str(path: str, data: dict[str, Any], key: str) -> str:
    if key not in data:
        raise SourceRegistryError(f"{path}: missing required field {key!r}")
    value = data[key]
    if not isinstance(value, str) or not value:
        raise SourceRegistryError(f"{path}.{key}: expected a non-empty string")
    return value


def _optional_str(path: str, data: dict[str, Any], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise SourceRegistryError(f"{path}.{key}: expected a non-empty string or null")
    return value


def _required_int(path: str, data: dict[str, Any], key: str) -> int:
    if key not in data:
        raise SourceRegistryError(f"{path}: missing required field {key!r}")
    value = data[key]
    if not isinstance(value, int) or value < 0:
        raise SourceRegistryError(f"{path}.{key}: expected a non-negative integer")
    return value


def _string_list(path: str, data: dict[str, Any], key: str) -> list[str]:
    value = data.get(key, [])
    if not isinstance(value, list) or any(not isinstance(x, str) or not x for x in value):
        raise SourceRegistryError(f"{path}.{key}: expected a list of non-empty strings")
    return value


def _parse_raw(path: str, data: dict[str, Any]) -> RawConfig:
    _reject_extra(path, data, {"root", "format", "path_globs", "manifest"})
    fmt = _required_str(path, data, "format")
    if fmt not in KNOWN_RAW_FORMATS:
        raise SourceRegistryError(f"{path}.format: expected one of {sorted(KNOWN_RAW_FORMATS)}")
    return RawConfig(
        root=_required_str(path, data, "root"),
        format=fmt,
        path_globs=_string_list(path, data, "path_globs"),
        manifest=_optional_str(path, data, "manifest"),
    )


def _parse_read(path: str, data: dict[str, Any]) -> ReadConfig:
    _reject_extra(path, data, {"text_field", "id_field", "metadata_fields"})
    return ReadConfig(
        text_field=_optional_str(path, data, "text_field"),
        id_field=_optional_str(path, data, "id_field"),
        metadata_fields=_string_list(path, data, "metadata_fields"),
    )


def _parse_language(path: str, data: dict[str, Any]) -> LanguageConfig:
    _reject_extra(path, data, {"mode", "value", "field", "known_values"})
    mode = _required_str(path, data, "mode")
    value = _optional_str(path, data, "value")
    field = _optional_str(path, data, "field")
    if mode not in KNOWN_LANGUAGE_MODES:
        raise SourceRegistryError(f"{path}.mode: expected one of {sorted(KNOWN_LANGUAGE_MODES)}")
    if mode == "fixed" and not value:
        raise SourceRegistryError(f"{path}.value: required when mode=fixed")
    if mode == "field" and not field:
        raise SourceRegistryError(f"{path}.field: required when mode=field")
    return LanguageConfig(
        mode=mode,
        value=value,
        field=field,
        known_values=_string_list(path, data, "known_values"),
    )


def _parse_cleaning(path: str, data: dict[str, Any]) -> CleaningConfig:
    _reject_extra(path, data, {"profile"})
    return CleaningConfig(profile=_required_str(path, data, "profile"))


def _parse_sampling(path: str, data: dict[str, Any]) -> SamplingConfig:
    _reject_extra(
        path,
        data,
        {
            "strategy",
            "default_train_docs_per_group",
            "default_val_docs_per_group",
            "split_strategy",
        },
    )
    strategy = _required_str(path, data, "strategy")
    split_strategy = _required_str(path, data, "split_strategy")
    if strategy not in KNOWN_SAMPLING_STRATEGIES:
        raise SourceRegistryError(
            f"{path}.strategy: expected one of {sorted(KNOWN_SAMPLING_STRATEGIES)}"
        )
    if split_strategy not in KNOWN_SPLIT_STRATEGIES:
        raise SourceRegistryError(
            f"{path}.split_strategy: expected one of {sorted(KNOWN_SPLIT_STRATEGIES)}"
        )
    return SamplingConfig(
        strategy=strategy,
        default_train_docs_per_group=_required_int(path, data, "default_train_docs_per_group"),
        default_val_docs_per_group=_required_int(path, data, "default_val_docs_per_group"),
        split_strategy=split_strategy,
    )


def _parse_source_spec(path: str, data: dict[str, Any]) -> SourceSpec:
    _reject_extra(
        path,
        data,
        {
            "schema_version",
            "source_id",
            "display_name",
            "status",
            "raw",
            "read",
            "default_bucket",
            "bucket_mapping",
            "language",
            "cleaning",
            "sampling",
            "known_issues",
        },
    )

    schema_version = _required_str(path, data, "schema_version")
    if schema_version != "source_registry_v1":
        raise SourceRegistryError(f"{path}.schema_version: expected 'source_registry_v1'")

    status = _required_str(path, data, "status")
    if status not in KNOWN_STATUSES:
        raise SourceRegistryError(f"{path}.status: expected one of {sorted(KNOWN_STATUSES)}")

    default_bucket = _required_str(path, data, "default_bucket")
    if default_bucket not in KNOWN_BUCKETS:
        raise SourceRegistryError(
            f"{path}.default_bucket: expected one of {sorted(KNOWN_BUCKETS)}"
        )

    raw = _parse_raw(f"{path}.raw", _required_mapping(path, data, "raw"))
    read = _parse_read(f"{path}.read", _required_mapping(path, data, "read"))
    if raw.format != "tree" and not read.text_field:
        raise SourceRegistryError(f"{path}.read.text_field: required unless raw.format=tree")

    bucket_mapping = data.get("bucket_mapping", {})
    if not isinstance(bucket_mapping, dict):
        raise SourceRegistryError(f"{path}.bucket_mapping: expected a mapping")

    known_issues = _string_list(path, data, "known_issues")
    if not known_issues:
        raise SourceRegistryError(f"{path}.known_issues: expected at least one entry")

    return SourceSpec(
        schema_version=schema_version,
        source_id=_required_str(path, data, "source_id"),
        display_name=_required_str(path, data, "display_name"),
        status=status,
        raw=raw,
        read=read,
        default_bucket=default_bucket,
        bucket_mapping=bucket_mapping,
        language=_parse_language(f"{path}.language", _required_mapping(path, data, "language")),
        cleaning=_parse_cleaning(f"{path}.cleaning", _required_mapping(path, data, "cleaning")),
        sampling=_parse_sampling(f"{path}.sampling", _required_mapping(path, data, "sampling")),
        known_issues=known_issues,
    )


def load_source_file(path: Path) -> SourceSpec:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise SourceRegistryError(f"{path}: invalid YAML: {exc}") from exc

    spec = _parse_source_spec(str(path), _mapping(str(path), data))

    expected_source_id = path.stem
    if spec.source_id != expected_source_id:
        raise SourceRegistryError(
            f"{path}: source_id={spec.source_id!r} must match filename stem {expected_source_id!r}"
        )

    return spec


def load_source_registry(config_dir: str | Path) -> dict[str, SourceSpec]:
    config_dir = Path(config_dir)
    if not config_dir.exists():
        raise SourceRegistryError(f"source registry directory does not exist: {config_dir}")
    if not config_dir.is_dir():
        raise SourceRegistryError(f"source registry path is not a directory: {config_dir}")

    registry: dict[str, SourceSpec] = {}
    for path in sorted(config_dir.glob("*.yaml")):
        spec = load_source_file(path)
        if spec.source_id in registry:
            raise SourceRegistryError(f"duplicate source_id: {spec.source_id}")
        registry[spec.source_id] = spec

    if not registry:
        raise SourceRegistryError(f"no source YAML files found in {config_dir}")

    return registry


def validate_source_registry(
    config_dir: str | Path,
    *,
    check_paths: bool = False,
) -> dict[str, Any]:
    registry = load_source_registry(config_dir)
    missing_roots = []

    if check_paths:
        for spec in registry.values():
            root = Path(spec.raw.root)
            if not root.exists():
                missing_roots.append(
                    {
                        "source_id": spec.source_id,
                        "root": spec.raw.root,
                    }
                )

    by_status: dict[str, int] = {}
    by_bucket: dict[str, int] = {}
    for spec in registry.values():
        by_status[spec.status] = by_status.get(spec.status, 0) + 1
        by_bucket[spec.default_bucket] = by_bucket.get(spec.default_bucket, 0) + 1

    summary = {
        "config_dir": str(config_dir),
        "sources": len(registry),
        "source_ids": sorted(registry),
        "by_status": dict(sorted(by_status.items())),
        "by_default_bucket": dict(sorted(by_bucket.items())),
        "check_paths": check_paths,
        "missing_roots": missing_roots,
    }

    if missing_roots:
        raise SourceRegistryError(f"missing raw roots: {missing_roots}")

    return summary
