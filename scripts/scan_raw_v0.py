#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from collections import defaultdict, Counter

import pyarrow.parquet as pq


PROJECT_ROOT = Path("/mnt/kai_kpfs/weilai/train/pt-data-pipeline")
RAW_ROOT = Path("/mnt/kai_kpfs/weilai/dataset/raw")

SOURCES = {
    "bigcode": {
        "path": RAW_ROOT / "bigcode",
        "bucket": "code",
    },
    "openwebmath": {
        "path": RAW_ROOT / "openwebmath",
        "bucket": "math",
    },
    "fineweb_edu_sample_350BT": {
        "path": RAW_ROOT / "fineweb_edu_sample_350BT",
        "bucket": "english_web",
    },
    "fineweb2_cmn_Hani": {
        "path": RAW_ROOT / "fineweb2_cmn_Hani",
        "bucket": "chinese_web",
    },
}

TEXT_FIELD_HINTS = {
    "text",
    "content",
    "contents",
    "code",
    "body",
    "markdown",
    "raw_content",
    "document",
    "article",
    "prompt",
    "completion",
    "response",
    "messages",
}


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


def guess_text_columns(columns):
    cols = []
    for c in columns:
        lc = str(c).lower()
        if lc in TEXT_FIELD_HINTS:
            cols.append(c)
        elif any(h in lc for h in ["text", "content", "code", "body", "markdown"]):
            cols.append(c)
    return cols


def inspect_parquet(path: Path):
    pf = pq.ParquetFile(path)
    columns = pf.schema.names
    return {
        "num_rows": pf.metadata.num_rows,
        "columns": columns,
        "text_columns_guess": guess_text_columns(columns),
    }


def inspect_json_like(path: Path, max_lines: int = 200):
    import gzip

    opener = gzip.open if path.name.lower().endswith(".gz") else open
    keys = Counter()
    rows = 0

    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows += 1
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    keys.update(obj.keys())
            except Exception:
                pass
            if rows >= max_lines:
                break

    columns = list(keys.keys())
    return {
        "num_rows_sampled": rows,
        "columns": columns,
        "text_columns_guess": guess_text_columns(columns),
    }


def inspect_txt(path: Path):
    return {
        "columns": ["text"],
        "text_columns_guess": ["text"],
    }


def inspect_file(path: Path, fmt: str):
    if fmt == "parquet":
        return inspect_parquet(path)
    if fmt in {"jsonl", "jsonl.gz", "json"}:
        return inspect_json_like(path)
    if fmt in {"txt", "txt.gz"}:
        return inspect_txt(path)
    return {}


def iter_files(root: Path):
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            yield Path(dirpath) / filename


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-files-per-source", type=int, default=0)
    parser.add_argument("--inspect-files-per-source", type=int, default=100)
    args = parser.parse_args()

    manifest_path = PROJECT_ROOT / "data/manifests/raw_files_v0.jsonl"
    report_path = PROJECT_ROOT / "data/reports/raw_scan_v0.md"

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    summary = defaultdict(lambda: {
        "files": 0,
        "bytes": 0,
        "formats": Counter(),
        "status": Counter(),
        "text_columns": Counter(),
        "errors": [],
    })

    with manifest_path.open("w", encoding="utf-8") as out:
        for source, meta in SOURCES.items():
            root = meta["path"]
            bucket = meta["bucket"]

            if not root.exists():
                row = {
                    "source": source,
                    "bucket": bucket,
                    "path": str(root),
                    "status": "missing_source_dir",
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                summary[source]["status"]["missing_source_dir"] += 1
                continue

            seen = 0
            inspected = 0

            for path in iter_files(root):
                seen += 1
                if args.max_files_per_source and seen > args.max_files_per_source:
                    break

                fmt = guess_format(path)
                size = path.stat().st_size

                row = {
                    "source": source,
                    "bucket": bucket,
                    "path": str(path),
                    "relpath": str(path.relative_to(root)),
                    "format": fmt,
                    "num_bytes": size,
                    "status": "ok",
                    "num_rows": None,
                    "num_rows_sampled": None,
                    "columns": [],
                    "text_columns_guess": [],
                    "error": None,
                }

                if inspected < args.inspect_files_per_source:
                    try:
                        info = inspect_file(path, fmt)
                        row.update(info)
                        inspected += 1
                    except Exception as e:
                        row["status"] = "inspect_error"
                        row["error"] = repr(e)

                out.write(json.dumps(row, ensure_ascii=False) + "\n")

                s = summary[source]
                s["files"] += 1
                s["bytes"] += size
                s["formats"][fmt] += 1
                s["status"][row["status"]] += 1

                for c in row.get("text_columns_guess") or []:
                    s["text_columns"][c] += 1

                if row["error"] and len(s["errors"]) < 20:
                    s["errors"].append({
                        "path": str(path),
                        "error": row["error"],
                    })

    with report_path.open("w", encoding="utf-8") as f:
        f.write("# Raw Dataset Scan v0\n\n")
        f.write(f"Manifest: `{manifest_path}`\n\n")

        for source, s in summary.items():
            f.write(f"## {source}\n\n")
            f.write(f"- files: {s['files']}\n")
            f.write(f"- bytes: {s['bytes']}\n")
            f.write(f"- approx GiB: {s['bytes'] / 1024**3:.2f}\n")
            f.write(f"- formats: {dict(s['formats'])}\n")
            f.write(f"- status: {dict(s['status'])}\n")
            f.write(f"- text column guesses: {dict(s['text_columns'].most_common(20))}\n")

            if s["errors"]:
                f.write("\n### Sample errors\n\n")
                for e in s["errors"]:
                    f.write(f"- `{e['path']}`: `{e['error']}`\n")

            f.write("\n")

    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote report:   {report_path}")


if __name__ == "__main__":
    main()
