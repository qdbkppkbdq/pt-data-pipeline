#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
from transformers import AutoTokenizer


PROJECT_ROOT = Path("/mnt/kai_kpfs/weilai/train/pt-data-pipeline")

DEFAULT_TRAIN = PROJECT_ROOT / "data/stage/proxy/proxy_train_v0.jsonl"
DEFAULT_VAL = PROJECT_ROOT / "data/stage/proxy/proxy_val_v0.jsonl"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data/stage/packed_glm_seq2048"


def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            r = json.loads(line)
            text = r.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            rows.append(r)
    return rows


def choose_eos_id(tokenizer, explicit_eos_id=None):
    if explicit_eos_id is not None:
        return int(explicit_eos_id)

    if tokenizer.eos_token_id is not None:
        return int(tokenizer.eos_token_id)

    # Some tokenizers define special tokens but no eos_token_id.
    for attr in ["sep_token_id", "pad_token_id"]:
        v = getattr(tokenizer, attr, None)
        if v is not None:
            return int(v)

    raise RuntimeError(
        "Cannot infer eos_token_id from tokenizer. "
        "Please pass --eos-id explicitly."
    )


def pack_split(
    *,
    rows,
    tokenizer,
    eos_id,
    seq_len,
    out_bin: Path,
    split_name: str,
    seed: int,
    shuffle_docs: bool,
    max_docs: int,
):
    stored_len = seq_len + 1
    rng = random.Random(seed)

    rows = list(rows)
    if shuffle_docs:
        rng.shuffle(rows)

    if max_docs > 0:
        rows = rows[:max_docs]

    out_bin.parent.mkdir(parents=True, exist_ok=True)

    # Write in append chunks as uint32.
    stats = {
        "split": split_name,
        "docs_seen": 0,
        "docs_packed": 0,
        "tokens_total_with_eos": 0,
        "tokens_written": 0,
        "tokens_dropped_remainder": 0,
        "num_sequences": 0,
        "seq_len": seq_len,
        "stored_len": stored_len,
        "eos_id": eos_id,
        "sources": Counter(),
        "buckets": Counter(),
        "token_len_by_source": defaultdict(list),
        "token_len_by_bucket": defaultdict(list),
    }

    buffer = []
    write_chunks = []

    with out_bin.open("wb") as f:
        for r in rows:
            stats["docs_seen"] += 1

            text = r["text"].strip()
            source = r.get("source", "unknown")
            bucket = r.get("bucket", "unknown")

            try:
                ids = tokenizer.encode(text, add_special_tokens=False)
            except Exception:
                continue

            if not ids:
                continue

            ids.append(eos_id)

            stats["docs_packed"] += 1
            stats["sources"][source] += 1
            stats["buckets"][bucket] += 1
            stats["tokens_total_with_eos"] += len(ids)
            stats["token_len_by_source"][source].append(len(ids))
            stats["token_len_by_bucket"][bucket].append(len(ids))

            buffer.extend(ids)

            while len(buffer) >= stored_len:
                seq = buffer[:stored_len]
                del buffer[:stored_len]

                arr = np.asarray(seq, dtype=np.uint32)
                arr.tofile(f)

                stats["num_sequences"] += 1
                stats["tokens_written"] += stored_len

    stats["tokens_dropped_remainder"] = len(buffer)

    # Convert non-json types.
    stats["sources"] = dict(stats["sources"])
    stats["buckets"] = dict(stats["buckets"])
    stats["token_len_by_source"] = {
        k: summarize_lens(v) for k, v in stats["token_len_by_source"].items()
    }
    stats["token_len_by_bucket"] = {
        k: summarize_lens(v) for k, v in stats["token_len_by_bucket"].items()
    }

    expected_bytes = stats["num_sequences"] * stored_len * np.dtype(np.uint32).itemsize
    actual_bytes = out_bin.stat().st_size if out_bin.exists() else 0

    stats["output_path"] = str(out_bin)
    stats["expected_bytes"] = expected_bytes
    stats["actual_bytes"] = actual_bytes
    stats["bytes_ok"] = expected_bytes == actual_bytes

    return stats


def percentile(xs, p):
    if not xs:
        return 0
    xs = sorted(xs)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    frac = k - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def summarize_lens(xs):
    if not xs:
        return {}
    return {
        "count": len(xs),
        "min": min(xs),
        "p50": percentile(xs, 50),
        "p90": percentile(xs, 90),
        "p95": percentile(xs, 95),
        "p99": percentile(xs, 99),
        "max": max(xs),
        "avg": sum(xs) / len(xs),
    }


def write_report(path: Path, meta):
    with path.open("w", encoding="utf-8") as out:
        out.write("# Packing Report v0\n\n")
        out.write(f"- tokenizer: `{meta['tokenizer_path']}`\n")
        out.write(f"- eos_id: `{meta['eos_id']}`\n")
        out.write(f"- seq_len: `{meta['seq_len']}`\n")
        out.write(f"- stored_len: `{meta['stored_len']}`\n")
        out.write(f"- dtype: `{meta['dtype']}`\n")
        out.write(f"- packing_method: `{meta['packing_method']}`\n")
        out.write(f"- shuffle_docs: `{meta['shuffle_docs']}`\n")
        out.write(f"- seed: `{meta['seed']}`\n\n")

        for split in ["train", "val"]:
            s = meta["splits"][split]
            out.write(f"## {split}\n\n")
            out.write(f"- output_path: `{s['output_path']}`\n")
            out.write(f"- docs_seen: {s['docs_seen']}\n")
            out.write(f"- docs_packed: {s['docs_packed']}\n")
            out.write(f"- tokens_total_with_eos: {s['tokens_total_with_eos']}\n")
            out.write(f"- num_sequences: {s['num_sequences']}\n")
            out.write(f"- tokens_written: {s['tokens_written']}\n")
            out.write(f"- tokens_dropped_remainder: {s['tokens_dropped_remainder']}\n")
            out.write(f"- actual_bytes: {s['actual_bytes']}\n")
            out.write(f"- bytes_ok: {s['bytes_ok']}\n")
            out.write(f"- sources: `{s['sources']}`\n")
            out.write(f"- buckets: `{s['buckets']}`\n\n")

            out.write("### token_len_by_source\n\n")
            for k, v in s["token_len_by_source"].items():
                out.write(f"- `{k}`: `{v}`\n")
            out.write("\n")

            out.write("### token_len_by_bucket\n\n")
            for k, v in s["token_len_by_bucket"].items():
                out.write(f"- `{k}`: `{v}`\n")
            out.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--val", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--eos-id", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-shuffle-docs", action="store_true")
    parser.add_argument("--max-train-docs", type=int, default=0)
    parser.add_argument("--max-val-docs", type=int, default=0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=True,
        local_files_only=True,
    )
    eos_id = choose_eos_id(tokenizer, args.eos_id)

    train_rows = read_jsonl(args.train)
    val_rows = read_jsonl(args.val)

    shuffle_docs = not args.no_shuffle_docs

    train_stats = pack_split(
        rows=train_rows,
        tokenizer=tokenizer,
        eos_id=eos_id,
        seq_len=args.seq_len,
        out_bin=args.out_dir / "train.bin",
        split_name="train",
        seed=args.seed,
        shuffle_docs=shuffle_docs,
        max_docs=args.max_train_docs,
    )

    val_stats = pack_split(
        rows=val_rows,
        tokenizer=tokenizer,
        eos_id=eos_id,
        seq_len=args.seq_len,
        out_bin=args.out_dir / "val.bin",
        split_name="val",
        seed=args.seed + 1,
        shuffle_docs=shuffle_docs,
        max_docs=args.max_val_docs,
    )

    meta = {
        "version": "packing_v0",
        "tokenizer_path": args.tokenizer,
        "tokenizer_class": tokenizer.__class__.__name__,
        "vocab_size": len(tokenizer),
        "eos_id": eos_id,
        "eos_token": tokenizer.eos_token,
        "pad_token_id": tokenizer.pad_token_id,
        "seq_len": args.seq_len,
        "stored_len": args.seq_len + 1,
        "dtype": "uint32",
        "packing_method": "shuffle_docs_then_concat_doc_eos_then_chunk",
        "long_docs": "split_across_chunks",
        "short_docs": "concatenate_until_full",
        "final_remainder": "drop",
        "cross_doc_attention_mask": False,
        "shuffle_docs": shuffle_docs,
        "seed": args.seed,
        "input_files": {
            "train": str(args.train),
            "val": str(args.val),
        },
        "splits": {
            "train": train_stats,
            "val": val_stats,
        },
    }

    meta_path = args.out_dir / "meta.json"
    report_path = args.out_dir / "pack_report_v0.md"

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    write_report(report_path, meta)

    print(f"Wrote train:  {args.out_dir / 'train.bin'}")
    print(f"Wrote val:    {args.out_dir / 'val.bin'}")
    print(f"Wrote meta:   {meta_path}")
    print(f"Wrote report: {report_path}")
    print()
    print("== train ==")
    print("docs_packed:", train_stats["docs_packed"])
    print("tokens_total_with_eos:", train_stats["tokens_total_with_eos"])
    print("num_sequences:", train_stats["num_sequences"])
    print("tokens_dropped_remainder:", train_stats["tokens_dropped_remainder"])
    print("bytes_ok:", train_stats["bytes_ok"])
    print()
    print("== val ==")
    print("docs_packed:", val_stats["docs_packed"])
    print("tokens_total_with_eos:", val_stats["tokens_total_with_eos"])
    print("num_sequences:", val_stats["num_sequences"])
    print("tokens_dropped_remainder:", val_stats["tokens_dropped_remainder"])
    print("bytes_ok:", val_stats["bytes_ok"])


if __name__ == "__main__":
    main()
