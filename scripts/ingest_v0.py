#!/usr/bin/env python3
"""
PT Data Pipeline - Ingest v0

This file implements the first stage of the PT data pipeline: INGEST.

It normalizes heterogeneous raw data sources into one JSONL schema.
It intentionally does NOT clean, deduplicate, score, or bucket records.
Those are later stages.

Supported v0 sources:
1. Hugging Face streaming datasets
   - HuggingFaceFW/fineweb
   - HuggingFaceFW/fineweb-2
   - open-web-math/open-web-math
2. Local repositories / directories
   - source files
   - tests
   - docs
   - configs

Install:
    pip install datasets pyarrow pyyaml tqdm rich

Examples:
    # FineWeb English sample
    python ingest_v0.py hf \
      --dataset HuggingFaceFW/fineweb \
      --name sample-10BT \
      --split train \
      --limit 10000 \
      --source fineweb_sample_10bt \
      --source-type web \
      --out data/stage/ingest/fineweb_10k.jsonl \
      --report data/reports/ingest_fineweb_10k.json

    # FineWeb2 Chinese cmn_Hani
    python ingest_v0.py hf \
      --dataset HuggingFaceFW/fineweb-2 \
      --name cmn_Hani \
      --split train \
      --limit 10000 \
      --source fineweb2_cmn_Hani \
      --source-type web \
      --out data/stage/ingest/fineweb2_cmn_Hani_10k.jsonl \
      --report data/reports/ingest_fineweb2_cmn_Hani_10k.json

    # OpenWebMath
    python ingest_v0.py hf \
      --dataset open-web-math/open-web-math \
      --split train \
      --limit 10000 \
      --source openwebmath \
      --source-type math_web \
      --out data/stage/ingest/openwebmath_10k.jsonl \
      --report data/reports/ingest_openwebmath_10k.json

    # Local repos
    python ingest_v0.py repos \
      --root data/raw/repos/cloned \
      --source local_repos \
      --out data/stage/ingest/repos.jsonl \
      --report data/reports/ingest_repos.json

Unified record schema:
    {
      "id": "...",
      "source": "fineweb2_cmn_Hani",
      "source_type": "web",
      "text": "...",
      "url": "...",
      "repo": null,
      "commit": null,
      "path": null,
      "license": null,
      "raw_ref": "...",
      "meta": {...},
      "quality": {},
      "risk": {},
      "tags": [],
      "bucket": null,
      "stage_allowed": ["PT"]
    }
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from hashlib import blake2b
from pathlib import Path
from typing import Any, Iterable, Iterator

try:
    from datasets import load_dataset
except ImportError:  # pragma: no cover
    load_dataset = None

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


# -----------------------------
# Schema
# -----------------------------


@dataclass
class DataRecord:
    id: str
    source: str
    source_type: str
    text: str

    # Provenance
    url: str | None = None
    repo: str | None = None
    commit: str | None = None
    path: str | None = None
    license: str | None = None
    raw_ref: str | None = None

    # Lightweight source metadata available at ingest time.
    meta: dict[str, Any] = field(default_factory=dict)

    # Empty containers for later pipeline stages.
    quality: dict[str, Any] = field(default_factory=dict)
    risk: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    # Filled later by bucket assignment.
    bucket: str | None = None
    stage_allowed: list[str] = field(default_factory=lambda: ["PT"])

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


@dataclass
class IngestStats:
    source: str
    source_type: str
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    seen: int = 0
    written: int = 0
    skipped_empty: int = 0
    skipped_too_large: int = 0
    skipped_decode_error: int = 0
    skipped_unsupported_file: int = 0
    text_bytes_written: int = 0
    languages: Counter = field(default_factory=Counter)
    file_types: Counter = field(default_factory=Counter)
    extensions: Counter = field(default_factory=Counter)
    sources: Counter = field(default_factory=Counter)
    errors: Counter = field(default_factory=Counter)

    def finish(self) -> None:
        self.finished_at = time.time()

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["duration_sec"] = round((self.finished_at or time.time()) - self.started_at, 3)
        out["languages"] = dict(self.languages)
        out["file_types"] = dict(self.file_types)
        out["extensions"] = dict(self.extensions)
        out["sources"] = dict(self.sources)
        out["errors"] = dict(self.errors)
        return out


# -----------------------------
# Utility functions
# -----------------------------


def stable_hash(*parts: str, prefix: str = "rec") -> str:
    h = blake2b(digest_size=16)
    sep = b"\x1f"
    for part in parts:
        h.update(part.encode("utf-8", errors="ignore"))
        h.update(sep)
    return f"{prefix}_{h.hexdigest()}"


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def short_preview(text: str, n: int = 120) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    return text[:n]


def iter_with_progress(iterable: Iterable[Any], total: int | None = None, desc: str = "") -> Iterable[Any]:
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc)


# -----------------------------
# Hugging Face ingest
# -----------------------------


def infer_text_field(row: dict[str, Any], preferred: str | None = None) -> str:
    if preferred and preferred in row:
        return safe_text(row[preferred])
    for key in ("text", "content", "document", "raw_content"):
        if key in row:
            return safe_text(row[key])
    return ""


def make_hf_record(
    row: dict[str, Any],
    source: str,
    source_type: str,
    text_field: str | None,
    row_index: int,
) -> DataRecord | None:
    text = infer_text_field(row, text_field).strip()
    if not text:
        return None

    url = row.get("url") or row.get("source_url") or row.get("warc_url")
    language = row.get("language") or row.get("lang")
    language_script = row.get("language_script")
    language_score = row.get("language_score")

    # Preserve lightweight metadata but avoid duplicating large text fields.
    meta: dict[str, Any] = {}
    for key in (
        "language",
        "language_script",
        "language_score",
        "date",
        "dump",
        "token_count",
        "word_count",
        "minhash_cluster_size",
        "score",
    ):
        if key in row and row[key] is not None:
            try:
                json.dumps(row[key])
                meta[key] = row[key]
            except TypeError:
                meta[key] = str(row[key])

    rec_id = stable_hash(source, str(url), str(row_index), text[:1000], prefix="hf")
    return DataRecord(
        id=rec_id,
        source=source,
        source_type=source_type,
        text=text,
        url=safe_text(url) if url else None,
        raw_ref=str(row_index),
        meta=meta,
    )


def ingest_hf(args: argparse.Namespace) -> None:
    if load_dataset is None:
        raise RuntimeError("datasets is not installed. Run: pip install datasets")

    ensure_parent(args.out)
    ensure_parent(args.report)

    stats = IngestStats(source=args.source, source_type=args.source_type)

    load_kwargs: dict[str, Any] = {
        "path": args.dataset,
        "split": args.split,
        "streaming": True,
    }
    if args.name:
        load_kwargs["name"] = args.name

    ds = load_dataset(**load_kwargs)

    with open(args.out, "w", encoding="utf-8") as out:
        iterator = iter(ds)
        if args.limit:
            iterator = _take(iterator, args.limit)
        for idx, row in enumerate(iter_with_progress(iterator, total=args.limit, desc=f"ingest {args.source}")):
            stats.seen += 1
            try:
                row = dict(row)
                rec = make_hf_record(row, args.source, args.source_type, args.text_field, idx)
                if rec is None:
                    stats.skipped_empty += 1
                    continue
                if args.max_chars and len(rec.text) > args.max_chars:
                    stats.skipped_too_large += 1
                    continue
                out.write(rec.to_json() + "\n")
                stats.written += 1
                stats.text_bytes_written += len(rec.text.encode("utf-8", errors="ignore"))
                if rec.meta.get("language"):
                    stats.languages[str(rec.meta["language"])] += 1
                elif rec.meta.get("language_script"):
                    stats.languages[str(rec.meta["language_script"])] += 1
                stats.sources[args.source] += 1
            except Exception as exc:  # pragma: no cover
                stats.errors[type(exc).__name__] += 1
                if args.fail_fast:
                    raise

    stats.finish()
    write_report(args.report, stats)


def _take(iterator: Iterator[Any], limit: int) -> Iterator[Any]:
    for i, item in enumerate(iterator):
        if i >= limit:
            break
        yield item


# -----------------------------
# Local repository ingest
# -----------------------------


SKIP_DIR_NAMES = {
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    "vendor",
    "dist",
    "build",
    "target",
    ".venv",
    "venv",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".tox",
    ".idea",
    ".vscode",
}

EXT_TO_LANGUAGE = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin",
    ".scala": "scala",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".php": "php",
    ".rb": "ruby",
    ".swift": "swift",
    ".sql": "sql",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".ps1": "powershell",
    ".md": "markdown",
    ".rst": "rst",
    ".txt": "text",
    ".json": "json",
    ".jsonl": "jsonl",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".ini": "ini",
    ".cfg": "ini",
    ".xml": "xml",
    ".html": "html",
    ".css": "css",
    ".dockerfile": "dockerfile",
}

CONFIG_NAMES = {
    "pyproject.toml",
    "package.json",
    "tsconfig.json",
    "setup.py",
    "setup.cfg",
    "requirements.txt",
    "cargo.toml",
    "go.mod",
    "pom.xml",
    "build.gradle",
    "dockerfile",
    "docker-compose.yml",
    "makefile",
}


def is_skipped_path(path: Path) -> bool:
    return any(part in SKIP_DIR_NAMES for part in path.parts)


def infer_language_from_path(path: Path) -> str | None:
    name = path.name.lower()
    if name == "dockerfile":
        return "dockerfile"
    if name == "makefile":
        return "makefile"
    return EXT_TO_LANGUAGE.get(path.suffix.lower())


def infer_file_type(path: Path, language: str | None) -> str:
    name = path.name.lower()
    parts = {p.lower() for p in path.parts}

    if name in CONFIG_NAMES or ".github" in parts:
        return "config"
    if name.startswith("readme"):
        return "readme"
    if name.startswith("license") or name in {"copying", "notice"}:
        return "license"
    if "test" in parts or "tests" in parts or re.search(r"(^test_|_test\.|\.test\.|\.spec\.)", name):
        return "test"
    if "doc" in parts or "docs" in parts or language in {"markdown", "rst"}:
        return "doc"
    if "example" in parts or "examples" in parts or "demo" in parts:
        return "example"
    if language in {"json", "yaml", "toml", "ini", "xml"}:
        return "config"
    return "source"


def iter_repo_files(root: Path) -> Iterator[tuple[str, Path, Path]]:
    """Yield (repo_name, repo_root, file_path). Assumes direct children are repos.

    If root itself looks like a repo or source tree, it is treated as one repo.
    """
    candidates = [p for p in root.iterdir() if p.is_dir()] if root.exists() else []
    direct_repo_mode = (root / ".git").exists() or any((root / name).exists() for name in ("pyproject.toml", "package.json", "go.mod", "Cargo.toml"))

    repo_roots: list[Path]
    if direct_repo_mode:
        repo_roots = [root]
    else:
        repo_roots = candidates

    for repo_root in repo_roots:
        repo_name = repo_root.name
        for fp in repo_root.rglob("*"):
            if not fp.is_file():
                continue
            rel = fp.relative_to(repo_root)
            if is_skipped_path(rel):
                continue
            yield repo_name, repo_root, fp


def read_text_file(path: Path, max_bytes: int) -> str | None:
    try:
        if path.stat().st_size > max_bytes:
            return None
        data = path.read_bytes()
        # Crude binary detection.
        if b"\x00" in data[:4096]:
            return None
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return None


def ingest_repos(args: argparse.Namespace) -> None:
    ensure_parent(args.out)
    ensure_parent(args.report)

    root = Path(args.root)
    stats = IngestStats(source=args.source, source_type="repo")

    files = list(iter_repo_files(root))
    if args.limit:
        files = files[: args.limit]

    with open(args.out, "w", encoding="utf-8") as out:
        for repo_name, repo_root, fp in iter_with_progress(files, total=len(files), desc="ingest repos"):
            stats.seen += 1
            rel_path = fp.relative_to(repo_root)
            lang = infer_language_from_path(rel_path)
            if lang is None:
                stats.skipped_unsupported_file += 1
                continue

            text = read_text_file(fp, args.max_file_bytes)
            if text is None:
                stats.skipped_decode_error += 1
                continue
            text = text.strip("\ufeff")
            if not text.strip():
                stats.skipped_empty += 1
                continue

            file_type = infer_file_type(rel_path, lang)
            ext = fp.suffix.lower() or fp.name.lower()

            rec = DataRecord(
                id=stable_hash(args.source, repo_name, str(rel_path), text[:1000], prefix="repo"),
                source=args.source,
                source_type="repo",
                text=text,
                repo=repo_name,
                path=str(rel_path),
                raw_ref=str(fp),
                meta={
                    "language": lang,
                    "file_type": file_type,
                    "extension": ext,
                    "repo_root": str(repo_root),
                    "size_bytes": fp.stat().st_size,
                },
            )
            out.write(rec.to_json() + "\n")
            stats.written += 1
            stats.text_bytes_written += len(text.encode("utf-8", errors="ignore"))
            stats.languages[lang] += 1
            stats.file_types[file_type] += 1
            stats.extensions[ext] += 1
            stats.sources[args.source] += 1

    stats.finish()
    write_report(args.report, stats)


# -----------------------------
# Reporting and validation
# -----------------------------


def write_report(path: str | Path, stats: IngestStats) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats.to_dict(), f, ensure_ascii=False, indent=2)


def validate_jsonl(args: argparse.Namespace) -> None:
    required = {"id", "source", "source_type", "text", "meta", "quality", "risk", "tags", "stage_allowed"}
    n = 0
    bad = 0
    ids: set[str] = set()
    dup_ids = 0
    source_counter = Counter()
    type_counter = Counter()

    with open(args.input, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            n += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                print(f"BAD JSON line={line_no}", file=sys.stderr)
                continue
            missing = required - set(rec)
            if missing:
                bad += 1
                print(f"MISSING {missing} line={line_no}", file=sys.stderr)
            if not isinstance(rec.get("text"), str) or not rec.get("text", "").strip():
                bad += 1
                print(f"EMPTY TEXT line={line_no}", file=sys.stderr)
            rid = rec.get("id")
            if rid in ids:
                dup_ids += 1
            ids.add(rid)
            source_counter[rec.get("source")] += 1
            type_counter[rec.get("source_type")] += 1

    result = {
        "records": n,
        "bad_records": bad,
        "duplicate_ids": dup_ids,
        "sources": dict(source_counter),
        "source_types": dict(type_counter),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.fail_on_bad and (bad > 0 or dup_ids > 0):
        raise SystemExit(1)


# -----------------------------
# CLI
# -----------------------------


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PT data pipeline ingest layer v0")
    sub = parser.add_subparsers(dest="cmd", required=True)

    hf = sub.add_parser("hf", help="ingest Hugging Face streaming dataset")
    hf.add_argument("--dataset", required=True, help="HF dataset path, e.g. HuggingFaceFW/fineweb-2")
    hf.add_argument("--name", default=None, help="HF dataset config/name, e.g. cmn_Hani or sample-10BT")
    hf.add_argument("--split", default="train")
    hf.add_argument("--limit", type=int, default=None, help="max records to read")
    hf.add_argument("--text-field", default=None)
    hf.add_argument("--source", required=True)
    hf.add_argument("--source-type", required=True, choices=["web", "math_web", "docs", "code", "repo", "other"])
    hf.add_argument("--out", required=True)
    hf.add_argument("--report", required=True)
    hf.add_argument("--max-chars", type=int, default=2_000_000)
    hf.add_argument("--fail-fast", action="store_true")
    hf.set_defaults(func=ingest_hf)

    repos = sub.add_parser("repos", help="ingest local repositories or source directories")
    repos.add_argument("--root", required=True)
    repos.add_argument("--source", default="local_repos")
    repos.add_argument("--out", required=True)
    repos.add_argument("--report", required=True)
    repos.add_argument("--limit", type=int, default=None, help="max files to scan")
    repos.add_argument("--max-file-bytes", type=int, default=2_000_000)
    repos.set_defaults(func=ingest_repos)

    val = sub.add_parser("validate", help="validate unified JSONL output")
    val.add_argument("--input", required=True)
    val.add_argument("--fail-on-bad", action="store_true")
    val.set_defaults(func=validate_jsonl)

    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
