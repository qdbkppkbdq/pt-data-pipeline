#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from collections import defaultdict, Counter

import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path("/mnt/kai_kpfs/weilai/train/pt-data-pipeline")
MANIFEST = PROJECT_ROOT / "data/manifests/raw_files_v0.jsonl"
REPORT = PROJECT_ROOT / "data/reports/raw_schema_probe_v0.md"


def suffix_key(path: str) -> str:
    p = Path(path)
    if p.suffixes:
        return "".join(p.suffixes).lower()
    return "<no_suffix>"


def magic_key(path: str) -> str:
    try:
        with open(path, "rb") as f:
            head = f.read(16)

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
    except Exception as e:
        return f"magic_error:{repr(e)}"


def is_stringy_arrow_type(t) -> bool:
    return (
        pa.types.is_string(t)
        or pa.types.is_large_string(t)
        or pa.types.is_binary(t)
        or pa.types.is_large_binary(t)
    )


def inspect_parquet(path: str, max_columns: int = 80):
    pf = pq.ParquetFile(path)
    schema = pf.schema_arrow

    columns = []
    stringy_columns = []

    for field in schema:
        columns.append({
            "name": field.name,
            "type": str(field.type),
        })
        if is_stringy_arrow_type(field.type):
            stringy_columns.append(field.name)

    sample_values = {}
    candidate_cols = stringy_columns[:max_columns]

    if candidate_cols and pf.metadata.num_row_groups > 0:
        table = pf.read_row_group(0, columns=candidate_cols)
        take_n = min(3, table.num_rows)

        for col in candidate_cols:
            vals = []
            arr = table[col]
            for i in range(take_n):
                try:
                    v = arr[i].as_py()
                    if isinstance(v, bytes):
                        v = v[:300].decode("utf-8", errors="replace")
                    elif isinstance(v, str):
                        v = v[:300].replace("\n", "\\n")
                    else:
                        v = repr(v)[:300]
                    vals.append(v)
                except Exception as e:
                    vals.append(f"<sample_error {repr(e)}>")
            sample_values[col] = vals

    return {
        "num_rows": pf.metadata.num_rows,
        "num_row_groups": pf.metadata.num_row_groups,
        "columns": columns,
        "stringy_columns": stringy_columns,
        "sample_values": sample_values,
    }


def should_try_parquet(fmt: str, magic: str) -> bool:
    # Trust magic more than suffix.
    # Some files have .parquet suffix but are not valid parquet.
    return magic == "parquet_magic"


def safe_inspect(path: str, fmt: str, magic: str):
    if should_try_parquet(fmt, magic):
        try:
            info = inspect_parquet(path)
            info["inspect_kind"] = "parquet"
            return info
        except Exception as e:
            return {
                "inspect_kind": "parquet_error",
                "inspect_error": repr(e),
            }

    return {
        "inspect_kind": "skipped",
        "inspect_reason": f"fmt={fmt}, magic={magic}",
    }


def load_manifest():
    rows = []
    with MANIFEST.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-files-per-group", type=int, default=3)
    args = parser.parse_args()

    rows = load_manifest()

    grouped = defaultdict(list)
    ext_counter = defaultdict(Counter)
    magic_counter = defaultdict(Counter)
    fmt_magic_counter = defaultdict(Counter)

    for r in rows:
        source = r.get("source", "unknown")
        path = r.get("path")
        fmt = r.get("format", "unknown")
        ext = suffix_key(path)
        magic = magic_key(path)

        ext_counter[source][ext] += 1
        magic_counter[source][magic] += 1
        fmt_magic_counter[source][f"fmt={fmt}, magic={magic}"] += 1

        key = (source, fmt, ext, magic)
        if len(grouped[key]) < args.max_files_per_group:
            grouped[key].append(r)

    REPORT.parent.mkdir(parents=True, exist_ok=True)

    with REPORT.open("w", encoding="utf-8") as out:
        out.write("# Raw Schema Probe v0\n\n")
        out.write(f"Manifest: `{MANIFEST}`\n\n")

        out.write("## Extension summary\n\n")
        for source, c in sorted(ext_counter.items()):
            out.write(f"### {source}\n\n")
            for k, v in c.most_common(40):
                out.write(f"- `{k}`: {v}\n")
            out.write("\n")

        out.write("## Magic summary\n\n")
        for source, c in sorted(magic_counter.items()):
            out.write(f"### {source}\n\n")
            for k, v in c.most_common(40):
                out.write(f"- `{k}`: {v}\n")
            out.write("\n")

        out.write("## Format x magic summary\n\n")
        for source, c in sorted(fmt_magic_counter.items()):
            out.write(f"### {source}\n\n")
            for k, v in c.most_common(60):
                out.write(f"- `{k}`: {v}\n")
            out.write("\n")

        out.write("## Schema samples\n\n")

        for key, items in sorted(grouped.items()):
            source, fmt, ext, magic = key
            out.write(f"### source={source}, fmt={fmt}, ext={ext}, magic={magic}\n\n")

            for r in items:
                path = r["path"]
                out.write(f"#### `{path}`\n\n")

                info = safe_inspect(path, fmt, magic)
                kind = info.get("inspect_kind")
                out.write(f"- inspect_kind: `{kind}`\n")

                if kind == "skipped":
                    out.write(f"- inspect_reason: `{info.get('inspect_reason')}`\n\n")
                    continue

                if "inspect_error" in info:
                    out.write(f"- inspect_error: `{info['inspect_error']}`\n\n")
                    continue

                out.write(f"- num_rows: {info.get('num_rows')}\n")
                out.write(f"- num_row_groups: {info.get('num_row_groups')}\n")
                out.write(f"- stringy_columns: `{info.get('stringy_columns')}`\n\n")

                out.write("Columns:\n\n")
                for col in info.get("columns", [])[:80]:
                    out.write(f"- `{col['name']}`: `{col['type']}`\n")

                out.write("\nSample values:\n\n")
                samples = info.get("sample_values", {})
                for col, vals in list(samples.items())[:30]:
                    out.write(f"- `{col}`:\n")
                    for v in vals:
                        out.write(f"  - `{v}`\n")

                out.write("\n")

    print(f"Wrote report: {REPORT}")


if __name__ == "__main__":
    main()
