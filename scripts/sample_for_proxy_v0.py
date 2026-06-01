#!/usr/bin/env python3
import argparse
import json
import random
import re
from pathlib import Path
from collections import defaultdict, Counter

import pyarrow.parquet as pq


PROJECT_ROOT = Path("/mnt/kai_kpfs/weilai/train/pt-data-pipeline")
MANIFEST = PROJECT_ROOT / "data/manifests/raw_files_v0.jsonl"

OUT_DIR = PROJECT_ROOT / "data/stage/proxy"
TRAIN_OUT = OUT_DIR / "proxy_train_v0.jsonl"
VAL_OUT = OUT_DIR / "proxy_val_v0.jsonl"
REPORT_OUT = PROJECT_ROOT / "data/reports/proxy_sample_v0.md"


SOURCE_CONFIG = {
    "openwebmath": {
        "bucket": "math",
        "text_col": "text",
    },
    "fineweb_edu_sample_350BT": {
        "bucket": "english_web",
        "text_col": "text",
    },
    "fineweb2_cmn_Hani": {
        "bucket": "chinese_web",
        "text_col": "text",
    },
}


SECRET_PATTERNS = [
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"(?i)(api[_-]?key|secret|token)\s*[:=]\s*['\"][A-Za-z0-9_\-]{16,}"),
]


def is_probably_bad_text(text: str, min_chars: int, max_chars: int):
    if text is None:
        return True, "none"
    if not isinstance(text, str):
        return True, "not_str"

    t = text.strip()

    if len(t) < min_chars:
        return True, "too_short"
    if len(t) > max_chars:
        return True, "too_long"
    if "\x00" in t:
        return True, "nul_byte"

    printable = sum(1 for ch in t if ch.isprintable())
    if printable / max(len(t), 1) < 0.80:
        return True, "low_printable"

    for pat in SECRET_PATTERNS:
        if pat.search(t):
            return True, "secret_like"

    return False, "ok"


def load_manifest():
    rows = []

    with MANIFEST.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            r = json.loads(line)
            source = r.get("source")

            if source not in SOURCE_CONFIG:
                continue

            path = Path(r.get("path"))

            # 这一轮只读真实 .parquet，跳过 HF cache / metadata / incomplete。
            if path.suffix != ".parquet":
                continue
            if ".cache" in path.parts:
                continue
            if not path.exists():
                continue

            rows.append(r)

    return rows


def stable_doc_id(source, path, row_group, row_index):
    return f"{source}:{path}:{row_group}:{row_index}"


def iter_parquet_texts(path: Path, text_col: str, max_row_groups: int, rng: random.Random):
    pf = pq.ParquetFile(path)
    nrg = pf.metadata.num_row_groups

    row_groups = list(range(nrg))
    rng.shuffle(row_groups)

    if max_row_groups > 0:
        row_groups = row_groups[:max_row_groups]

    for rg in row_groups:
        try:
            table = pf.read_row_group(rg, columns=[text_col])
        except Exception:
            continue

        col = table[text_col]
        indices = list(range(len(col)))
        rng.shuffle(indices)

        for i in indices:
            try:
                yield rg, i, col[i].as_py()
            except Exception:
                continue


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-docs-per-source", type=int, default=2000)
    parser.add_argument("--val-docs-per-source", type=int, default=200)
    parser.add_argument("--max-files-per-source", type=int, default=8)
    parser.add_argument("--max-row-groups-per-file", type=int, default=2)
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument("--max-chars", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)

    manifest_rows = load_manifest()

    rows_by_source = defaultdict(list)
    for r in manifest_rows:
        rows_by_source[r["source"]].append(r)

    for source in rows_by_source:
        rng.shuffle(rows_by_source[source])
        rows_by_source[source] = rows_by_source[source][:args.max_files_per_source]

    stats = defaultdict(Counter)
    examples = defaultdict(list)

    with TRAIN_OUT.open("w", encoding="utf-8") as train_f, VAL_OUT.open("w", encoding="utf-8") as val_f:
        for source in sorted(SOURCE_CONFIG):
            rows = rows_by_source.get(source, [])
            cfg = SOURCE_CONFIG[source]
            bucket = cfg["bucket"]
            text_col = cfg["text_col"]

            target_train = args.target_docs_per_source
            target_val = args.val_docs_per_source
            target_total = target_train + target_val

            accepted = 0

            for r in rows:
                path = Path(r["path"])
                stats[source]["files_seen"] += 1

                for row_group, row_index, text in iter_parquet_texts(
                    path=path,
                    text_col=text_col,
                    max_row_groups=args.max_row_groups_per_file,
                    rng=rng,
                ):
                    bad, reason = is_probably_bad_text(
                        text,
                        min_chars=args.min_chars,
                        max_chars=args.max_chars,
                    )

                    if bad:
                        stats[source][f"reject_{reason}"] += 1
                        continue

                    split = "val" if accepted < target_val else "train"

                    text = text.strip()
                    row = {
                        "id": stable_doc_id(source, str(path), row_group, row_index),
                        "source": source,
                        "bucket": bucket,
                        "text": text,
                        "meta": {
                            "path": str(path),
                            "row_group": row_group,
                            "row_index_in_row_group": row_index,
                        },
                    }

                    if split == "val":
                        val_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    else:
                        train_f.write(json.dumps(row, ensure_ascii=False) + "\n")

                    accepted += 1
                    stats[source][f"accepted_{split}"] += 1
                    stats[source]["accepted_total"] += 1
                    stats[source]["chars_total"] += len(text)

                    if len(examples[source]) < 3:
                        examples[source].append(text[:500].replace("\n", "\\n"))

                    if accepted >= target_total:
                        break

                if accepted >= target_total:
                    break

            if accepted < target_total:
                stats[source]["under_target"] = target_total - accepted

    with REPORT_OUT.open("w", encoding="utf-8") as out:
        out.write("# Proxy Sample v0 Report\n\n")
        out.write(f"- train: `{TRAIN_OUT}`\n")
        out.write(f"- val: `{VAL_OUT}`\n\n")

        for source in sorted(SOURCE_CONFIG):
            out.write(f"## {source}\n\n")
            out.write(f"- bucket: `{SOURCE_CONFIG[source]['bucket']}`\n")
            out.write(f"- text_col: `{SOURCE_CONFIG[source]['text_col']}`\n")

            for k, v in stats[source].most_common():
                out.write(f"- {k}: {v}\n")

            out.write("\n### Examples\n\n")
            for ex in examples[source]:
                out.write(f"- `{ex}`\n")

            out.write("\n")

    print(f"Wrote train:  {TRAIN_OUT}")
    print(f"Wrote val:    {VAL_OUT}")
    print(f"Wrote report: {REPORT_OUT}")


if __name__ == "__main__":
    main()
