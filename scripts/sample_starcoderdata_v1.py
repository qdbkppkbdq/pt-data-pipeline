#!/usr/bin/env python3
import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq


GENERATED_PATTERNS = [
    "node_modules/", "vendor/", "third_party/", "third-party/",
    "dist/", "build/", "target/", ".min.js", ".bundle.js",
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
]

SECRET_PATTERNS = [
    re.compile(r"-----BEGIN (RSA |DSA |EC |OPENSSH |PGP )?PRIVATE KEY-----"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"(?i)(api[_-]?key|secret|password|passwd|token)\s*[:=]\s*['\"][^'\"]{16,}['\"]"),
]

def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

def reject_reason(text: str, path: str, min_chars: int, max_chars: int):
    if not isinstance(text, str):
        return "not_string"
    if not text.strip():
        return "empty"
    n = len(text)
    if n < min_chars:
        return "too_short"
    if n > max_chars:
        return "too_long"
    if "\x00" in text:
        return "nul_char"
    nonprint = sum(1 for ch in text if ord(ch) < 32 and ch not in "\n\r\t")
    if n and nonprint / n > 0.01:
        return "too_many_control_chars"
    lines = text.splitlines()
    if lines:
        max_line = max(len(x) for x in lines)
        if max_line > 20000:
            return "very_long_line"
    lower_path = (path or "").lower()
    if any(p in lower_path for p in GENERATED_PATTERNS):
        return "generated_or_vendor_path"
    for pat in SECRET_PATTERNS:
        if pat.search(text):
            return "secret_like"
    return None

def iter_parquet_rows(files, columns):
    for f in files:
        try:
            pf = pq.ParquetFile(f)
        except Exception as e:
            yield ("__FILE_ERROR__", str(f), repr(e))
            continue

        for rg in range(pf.num_row_groups):
            try:
                table = pf.read_row_group(rg, columns=columns)
            except Exception as e:
                yield ("__ROWGROUP_ERROR__", str(f), repr(e))
                continue

            for row in table.to_pylist():
                yield row, str(f), rg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--languages", required=True, help="comma separated")
    ap.add_argument("--docs-per-lang", type=int, default=2000)
    ap.add_argument("--val-docs-per-lang", type=int, default=200)
    ap.add_argument("--max-files-per-lang", type=int, default=8)
    ap.add_argument("--min-chars", type=int, default=80)
    ap.add_argument("--max-chars", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-out", required=True)
    ap.add_argument("--val-out", required=True)
    ap.add_argument("--report-out", required=True)
    args = ap.parse_args()

    random.seed(args.seed)

    root = Path(args.root)
    langs = [x.strip() for x in args.languages.split(",") if x.strip()]

    train_out = Path(args.train_out)
    val_out = Path(args.val_out)
    report_out = Path(args.report_out)
    train_out.parent.mkdir(parents=True, exist_ok=True)
    val_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.parent.mkdir(parents=True, exist_ok=True)

    total_needed = args.docs_per_lang + args.val_docs_per_lang

    seen_norm_hash = set()
    accepted_by_lang = defaultdict(list)
    reject_counts = Counter()
    input_counts = Counter()
    file_counts = Counter()

    columns = ["max_stars_repo_path", "max_stars_repo_name", "max_stars_count", "id", "content"]

    for lang in langs:
        files = sorted((root / lang).rglob("*.parquet"))
        files = [p for p in files if ".cache" not in p.parts]
        file_counts[lang] = len(files)

        if not files:
            reject_counts[(lang, "no_parquet")] += 1
            continue

        # spread across available shards, but keep probe bounded
        files = files[: args.max_files_per_lang]

        for row, raw_file, rg in iter_parquet_rows(files, columns):
            if row == "__FILE_ERROR__":
                reject_counts[(lang, "file_error")] += 1
                continue
            if row == "__ROWGROUP_ERROR__":
                reject_counts[(lang, "rowgroup_error")] += 1
                continue

            input_counts[lang] += 1

            content = row.get("content")
            repo_path = row.get("max_stars_repo_path") or ""
            reason = reject_reason(content, repo_path, args.min_chars, args.max_chars)
            if reason:
                reject_counts[(lang, reason)] += 1
                continue

            h = sha1(norm_text(content))
            if h in seen_norm_hash:
                reject_counts[(lang, "exact_norm_duplicate")] += 1
                continue
            seen_norm_hash.add(h)

            item = {
                "id": f"starcoderdata:{lang}:{row.get('id')}",
                "source": "starcoderdata",
                "bucket": "code" if lang != "markdown" else "docs",
                "language": lang,
                "text": content,
                "meta": {
                    "raw_file": raw_file,
                    "row_group": rg,
                    "raw_id": row.get("id"),
                    "repo_path": repo_path,
                    "repo_name": row.get("max_stars_repo_name"),
                    "stars": row.get("max_stars_count"),
                    "content_sha1": sha1(content),
                    "norm_sha1": h,
                    "clean_version": "code_clean_v1_basic",
                },
            }
            accepted_by_lang[lang].append(item)

            if len(accepted_by_lang[lang]) >= total_needed:
                break

    train_items = []
    val_items = []
    for lang, items in accepted_by_lang.items():
        random.shuffle(items)
        val_items.extend(items[: args.val_docs_per_lang])
        train_items.extend(items[args.val_docs_per_lang : args.val_docs_per_lang + args.docs_per_lang])

    random.shuffle(train_items)
    random.shuffle(val_items)

    with train_out.open("w", encoding="utf-8") as f:
        for x in train_items:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    with val_out.open("w", encoding="utf-8") as f:
        for x in val_items:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    lines = []
    lines.append("# Starcoderdata Code Sample v1")
    lines.append("")
    lines.append(f"- root: `{root}`")
    lines.append(f"- languages: `{langs}`")
    lines.append(f"- docs_per_lang: `{args.docs_per_lang}`")
    lines.append(f"- val_docs_per_lang: `{args.val_docs_per_lang}`")
    lines.append(f"- max_files_per_lang: `{args.max_files_per_lang}`")
    lines.append(f"- train_out: `{train_out}`")
    lines.append(f"- val_out: `{val_out}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- train_docs: {len(train_items)}")
    lines.append(f"- val_docs: {len(val_items)}")
    lines.append(f"- unique_norm_hashes: {len(seen_norm_hash)}")
    lines.append("")
    lines.append("## By language")
    lines.append("")
    lines.append("| language | parquet_files_seen | input_rows | accepted_total | train | val |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for lang in langs:
        accepted = len(accepted_by_lang.get(lang, []))
        train_n = sum(1 for x in train_items if x["language"] == lang)
        val_n = sum(1 for x in val_items if x["language"] == lang)
        lines.append(f"| {lang} | {file_counts[lang]} | {input_counts[lang]} | {accepted} | {train_n} | {val_n} |")

    lines.append("")
    lines.append("## Reject reasons")
    lines.append("")
    lines.append("| language | reason | count |")
    lines.append("|---|---|---:|")
    for (lang, reason), count in reject_counts.most_common():
        lines.append(f"| {lang} | {reason} | {count} |")

    report_out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {train_out} docs={len(train_items)}")
    print(f"wrote {val_out} docs={len(val_items)}")
    print(f"wrote {report_out}")

if __name__ == "__main__":
    main()
