#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.sources import SourceSpec, load_source_registry  # noqa: E402


MANIFEST_SCHEMA_VERSION = "manifest_v1"
SUPPORTED_INSPECT_FORMATS = {"parquet", "jsonl", "txt", "tree"}


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def guess_format(path: Path) -> str:
    name = path.name.lower()
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        return "parquet"
    if suffix in {".jsonl", ".ndjson"}:
        return "jsonl"
    if suffix == ".json":
        return "json"
    if suffix in {".txt", ".text"}:
        return "txt"
    if suffix in {".arrow", ".feather"}:
        return "arrow"
    if name.endswith(".jsonl.gz"):
        return "jsonl.gz"
    if name.endswith(".txt.gz"):
        return "txt.gz"
    return "unknown"


def magic_key(path: Path) -> str:
    try:
        if path.is_dir():
            return "directory"
        with path.open("rb") as f:
            head = f.read(16)
    except Exception as exc:
        return f"magic_error:{exc!r}"

    if head.startswith(b"PAR1"):
        return "parquet_magic"
    if head.startswith(b"\x1f\x8b"):
        return "gzip_magic"
    if head.startswith(b"\x28\xb5\x2f\xfd"):
        return "zstd_magic"
    if head[:1] in {b"{", b"["}:
        return "json_like_magic"
    if not head:
        return "empty"
    return head[:8].hex()


def detect_format(path: Path, magic: str) -> str:
    if path.is_dir():
        return "tree"
    if magic == "parquet_magic":
        return "parquet"
    if magic == "gzip_magic":
        name = path.name.lower()
        if name.endswith(".jsonl.gz"):
            return "jsonl.gz"
        if name.endswith(".txt.gz"):
            return "txt.gz"
        return "gzip"
    if magic == "json_like_magic":
        guessed = guess_format(path)
        return guessed if guessed in {"json", "jsonl"} else "json_like"
    return guess_format(path)


def resolve_output_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def iter_source_paths(spec: SourceSpec) -> list[Path]:
    root = Path(spec.raw.root)
    if spec.raw.format == "tree" and not spec.raw.path_globs:
        return [root]

    paths: list[Path] = []
    seen: set[Path] = set()
    for pattern in spec.raw.path_globs:
        for path in sorted(root.glob(pattern)):
            if path in seen:
                continue
            seen.add(path)
            paths.append(path)
    return paths


def infer_language(spec: SourceSpec, path: Path, relpath: str) -> str | None:
    lang_cfg = spec.language
    if lang_cfg.mode == "fixed":
        return lang_cfg.value
    if lang_cfg.mode == "field":
        if len(lang_cfg.known_values) == 1:
            return lang_cfg.known_values[0]
        return None
    if lang_cfg.mode == "path_parent":
        rel = Path(relpath)
        parent = rel.parent.name
        if parent and parent != ".":
            return parent
        return path.parent.name or None
    return None


def source_level_language(spec: SourceSpec) -> str | None:
    if spec.language.mode == "fixed":
        return spec.language.value
    if len(spec.language.known_values) == 1:
        return spec.language.known_values[0]
    return None


def infer_bucket(spec: SourceSpec, language: str | None) -> str:
    mapping = spec.bucket_mapping.get("language")
    if isinstance(mapping, dict):
        if language in mapping:
            return str(mapping[language])
        if "default" in mapping:
            return str(mapping["default"])
    return spec.default_bucket


def inspect_parquet(path: Path) -> dict[str, Any]:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    schema = pf.schema_arrow
    columns = [{"name": field.name, "type": str(field.type)} for field in schema]
    return {
        "num_rows": pf.metadata.num_rows,
        "num_row_groups": pf.metadata.num_row_groups,
        "columns": columns,
    }


def inspect_basic(path: Path, spec: SourceSpec, detected_format: str, magic: str) -> tuple[str, str | None, dict[str, Any]]:
    if spec.raw.format not in SUPPORTED_INSPECT_FORMATS:
        return "unsupported_format", None, {}

    if spec.raw.format == "parquet":
        if magic != "parquet_magic":
            return "unsupported_format", f"expected parquet magic, got {magic}", {}
        try:
            return "ok", None, inspect_parquet(path)
        except Exception as exc:
            return "inspect_error", repr(exc), {}

    if spec.raw.format == "jsonl":
        if detected_format not in {"jsonl", "jsonl.gz", "json_like"}:
            return "unsupported_format", f"expected jsonl, detected {detected_format}", {}
        return "ok", None, {}

    if spec.raw.format == "txt":
        if detected_format not in {"txt", "txt.gz", "unknown"}:
            return "unsupported_format", f"expected txt, detected {detected_format}", {}
        return "ok", None, {}

    if spec.raw.format == "tree":
        return "ok", None, {}

    return "unsupported_format", f"unsupported raw format {spec.raw.format}", {}


def trainability_reason(
    *,
    spec: SourceSpec,
    status: str,
    num_bytes: int | None,
    columns: list[dict[str, str]],
) -> tuple[bool, str]:
    if spec.status != "usable":
        return False, f"source_{spec.status}"
    if status != "ok":
        return False, status
    if num_bytes is not None and num_bytes == 0:
        return False, "empty_file"

    if spec.raw.format == "parquet":
        column_names = {c["name"] for c in columns}
        if not spec.read.text_field or spec.read.text_field not in column_names:
            return False, "missing_text_field"
        return True, "ok"

    if spec.raw.format == "jsonl":
        if not spec.read.text_field:
            return False, "missing_text_field"
        return False, "schema_unverified"

    if spec.raw.format in {"txt", "tree"}:
        return True, "ok"

    return False, "unsupported_format"


def source_level_row(spec: SourceSpec, status: str, trainable_reason: str, probed_at: str) -> dict[str, Any]:
    language = source_level_language(spec)
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "source_id": spec.source_id,
        "source_status": spec.status,
        "bucket": infer_bucket(spec, language),
        "language": language,
        "path": spec.raw.root,
        "relpath": "",
        "format": spec.raw.format,
        "detected_format": None,
        "magic": None,
        "num_bytes": None,
        "num_rows": None,
        "num_row_groups": None,
        "columns": [],
        "text_field": spec.read.text_field,
        "id_field": spec.read.id_field,
        "trainable": False,
        "trainable_reason": trainable_reason,
        "status": status,
        "error": None,
        "probed_at": probed_at,
    }


def build_row(spec: SourceSpec, path: Path, probed_at: str) -> dict[str, Any]:
    root = Path(spec.raw.root)
    relpath = str(path.relative_to(root)) if path != root else ""
    language = infer_language(spec, path, relpath)
    bucket = infer_bucket(spec, language)
    magic = magic_key(path)
    detected_format = detect_format(path, magic)

    try:
        num_bytes = path.stat().st_size if path.is_file() else None
    except Exception:
        num_bytes = None

    status, error, inspected = inspect_basic(path, spec, detected_format, magic)
    columns = inspected.get("columns", [])
    trainable, trainable_reason = trainability_reason(
        spec=spec,
        status=status,
        num_bytes=num_bytes,
        columns=columns,
    )

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "source_id": spec.source_id,
        "source_status": spec.status,
        "bucket": bucket,
        "language": language,
        "path": str(path),
        "relpath": relpath,
        "format": spec.raw.format,
        "detected_format": detected_format,
        "magic": magic,
        "num_bytes": num_bytes,
        "num_rows": inspected.get("num_rows"),
        "num_row_groups": inspected.get("num_row_groups"),
        "columns": columns,
        "text_field": spec.read.text_field,
        "id_field": spec.read.id_field,
        "trainable": trainable,
        "trainable_reason": trainable_reason,
        "status": status,
        "error": error,
        "probed_at": probed_at,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_source: dict[str, Any] = {}
    for row in rows:
        source_id = row["source_id"]
        stats = by_source.setdefault(
            source_id,
            {
                "files": 0,
                "bytes": 0,
                "rows": 0,
                "row_groups": 0,
                "status": Counter(),
                "trainable": Counter(),
                "trainable_reason": Counter(),
                "bucket": Counter(),
                "language": Counter(),
            },
        )
        stats["files"] += 1
        stats["bytes"] += row.get("num_bytes") or 0
        stats["rows"] += row.get("num_rows") or 0
        stats["row_groups"] += row.get("num_row_groups") or 0
        stats["status"][row["status"]] += 1
        stats["trainable"][str(row["trainable"]).lower()] += 1
        stats["trainable_reason"][row["trainable_reason"]] += 1
        stats["bucket"][row["bucket"]] += 1
        if row.get("language"):
            stats["language"][row["language"]] += 1

    return by_source


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False, default=json_default) + "\n")


def write_report(path: Path, rows: list[dict[str, Any]], out_dir: Path) -> None:
    summary = summarize(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        out.write("# Manifest v1 Report\n\n")
        out.write(f"- schema_version: `{MANIFEST_SCHEMA_VERSION}`\n")
        out.write(f"- all_sources: `{out_dir / 'all_sources.jsonl'}`\n")
        out.write(f"- sources: {len(summary)}\n")
        out.write(f"- files: {len(rows)}\n\n")

        for source_id in sorted(summary):
            stats = summary[source_id]
            out.write(f"## {source_id}\n\n")
            out.write(f"- files: {stats['files']}\n")
            out.write(f"- bytes: {stats['bytes']}\n")
            out.write(f"- approx GiB: {stats['bytes'] / 1024**3:.2f}\n")
            out.write(f"- rows: {stats['rows']}\n")
            out.write(f"- row_groups: {stats['row_groups']}\n")
            out.write(f"- status: {dict(stats['status'])}\n")
            out.write(f"- trainable: {dict(stats['trainable'])}\n")
            out.write(f"- trainable_reason: {dict(stats['trainable_reason'])}\n")
            out.write(f"- buckets: {dict(stats['bucket'])}\n")
            if stats["language"]:
                out.write(f"- languages: {dict(stats['language'].most_common(30))}\n")
            out.write(f"- manifest: `{out_dir / f'{source_id}.jsonl'}`\n\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a file-level Manifest v1 from Source Registry YAML files."
    )
    parser.add_argument("--config-dir", default="configs/sources")
    parser.add_argument("--out-dir", default="data/manifests/source_registry_v1")
    parser.add_argument("--report", default="data/reports/manifest_v1.md")
    parser.add_argument("--max-files-per-source", type=int, default=0)
    args = parser.parse_args()

    registry = load_source_registry(args.config_dir)
    out_dir = resolve_output_path(args.out_dir)
    report_path = resolve_output_path(args.report)
    probed_at = datetime.now(UTC).replace(microsecond=0).isoformat()

    rows: list[dict[str, Any]] = []
    rows_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for source_id, spec in sorted(registry.items()):
        root = Path(spec.raw.root)
        if not root.exists():
            row = source_level_row(spec, "missing_root", "missing_root", probed_at)
            rows.append(row)
            rows_by_source[source_id].append(row)
            continue

        paths = iter_source_paths(spec)
        if args.max_files_per_source > 0:
            paths = paths[: args.max_files_per_source]

        if not paths:
            row = source_level_row(spec, "no_matching_files", "no_matching_files", probed_at)
            rows.append(row)
            rows_by_source[source_id].append(row)
            continue

        for path in paths:
            row = build_row(spec, path, probed_at)
            rows.append(row)
            rows_by_source[source_id].append(row)

    write_jsonl(out_dir / "all_sources.jsonl", rows)
    for source_id, source_rows in sorted(rows_by_source.items()):
        write_jsonl(out_dir / f"{source_id}.jsonl", source_rows)
    write_report(report_path, rows, out_dir)

    print(
        json.dumps(
            {
                "ok": True,
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "sources": len(rows_by_source),
                "files": len(rows),
                "out_dir": str(out_dir),
                "report": str(report_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
