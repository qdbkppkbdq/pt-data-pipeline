#!/usr/bin/env python3
import argparse
import json
import math
import statistics
from pathlib import Path
from collections import defaultdict, Counter


PROJECT_ROOT = Path("/mnt/kai_kpfs/weilai/train/pt-data-pipeline")

DEFAULT_TRAIN = PROJECT_ROOT / "data/stage/proxy/proxy_train_v0.jsonl"
DEFAULT_VAL = PROJECT_ROOT / "data/stage/proxy/proxy_val_v0.jsonl"
DEFAULT_REPORT = PROJECT_ROOT / "data/reports/token_stats_v0.md"
DEFAULT_JSON = PROJECT_ROOT / "data/reports/token_stats_v0.json"


def percentile(values, p):
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return xs[int(k)]
    return xs[f] * (c - k) + xs[c] * (k - f)


class ByteTokenizer:
    name = "utf8_byte_fallback"

    def encode_len(self, text: str) -> int:
        # Not a real LM tokenizer.
        # This is a conservative fallback for pipeline sizing only.
        return len(text.encode("utf-8"))


class TransformersTokenizer:
    def __init__(self, tokenizer_path: str):
        from transformers import AutoTokenizer

        self.tokenizer_path = tokenizer_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        self.name = f"transformers:{tokenizer_path}"

    def encode_len(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))


def build_tokenizer(args):
    if args.tokenizer:
        try:
            return TransformersTokenizer(args.tokenizer)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load tokenizer via transformers from {args.tokenizer!r}. "
                f"Make sure the tokenizer exists locally. Original error: {repr(e)}"
            )

    if args.allow_byte_fallback:
        return ByteTokenizer()

    raise RuntimeError(
        "No tokenizer provided. Pass --tokenizer /path/to/local/tokenizer "
        "or use --allow-byte-fallback for rough sizing only."
    )


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                yield line_no, json.loads(line)
            except Exception as e:
                raise RuntimeError(f"Bad JSON at {path}:{line_no}: {repr(e)}")


def init_group():
    return {
        "docs": 0,
        "chars": 0,
        "tokens": 0,
        "token_lens": [],
        "char_lens": [],
    }


def update_group(g, char_len, tok_len):
    g["docs"] += 1
    g["chars"] += char_len
    g["tokens"] += tok_len
    g["token_lens"].append(tok_len)
    g["char_lens"].append(char_len)


def summarize_group(g, seq_lens):
    docs = g["docs"]
    tokens = g["tokens"]
    chars = g["chars"]
    token_lens = g["token_lens"]
    char_lens = g["char_lens"]

    out = {
        "docs": docs,
        "chars": chars,
        "tokens": tokens,
        "avg_chars_per_doc": chars / docs if docs else 0,
        "avg_tokens_per_doc": tokens / docs if docs else 0,
        "chars_per_token": chars / tokens if tokens else 0,
        "token_len_min": min(token_lens) if token_lens else 0,
        "token_len_p50": percentile(token_lens, 50),
        "token_len_p90": percentile(token_lens, 90),
        "token_len_p95": percentile(token_lens, 95),
        "token_len_p99": percentile(token_lens, 99),
        "token_len_max": max(token_lens) if token_lens else 0,
        "char_len_p50": percentile(char_lens, 50),
        "char_len_p95": percentile(char_lens, 95),
        "char_len_max": max(char_lens) if char_lens else 0,
        "seq_estimates": {},
    }

    for seq_len in seq_lens:
        full_sequences = tokens // seq_len
        leftover_tokens = tokens % seq_len
        out["seq_estimates"][str(seq_len)] = {
            "full_sequences_if_concatenated": full_sequences,
            "leftover_tokens": leftover_tokens,
            "tokens_coverage": full_sequences * seq_len,
        }

    return out


def analyze_file(path: Path, tokenizer, seq_lens, max_docs=0):
    groups = {
        "all": init_group(),
        "by_source": defaultdict(init_group),
        "by_bucket": defaultdict(init_group),
    }

    bad = Counter()
    examples = defaultdict(list)

    for idx, (_, row) in enumerate(iter_jsonl(path), start=1):
        if max_docs and idx > max_docs:
            break

        text = row.get("text")
        source = row.get("source", "unknown")
        bucket = row.get("bucket", "unknown")

        if not isinstance(text, str) or not text:
            bad["missing_text"] += 1
            continue

        char_len = len(text)
        try:
            tok_len = tokenizer.encode_len(text)
        except Exception:
            bad["tokenize_error"] += 1
            continue

        update_group(groups["all"], char_len, tok_len)
        update_group(groups["by_source"][source], char_len, tok_len)
        update_group(groups["by_bucket"][bucket], char_len, tok_len)

        if len(examples[source]) < 2:
            examples[source].append({
                "chars": char_len,
                "tokens": tok_len,
                "preview": text[:300].replace("\n", "\\n"),
            })

    result = {
        "path": str(path),
        "bad": dict(bad),
        "all": summarize_group(groups["all"], seq_lens),
        "by_source": {
            k: summarize_group(v, seq_lens)
            for k, v in sorted(groups["by_source"].items())
        },
        "by_bucket": {
            k: summarize_group(v, seq_lens)
            for k, v in sorted(groups["by_bucket"].items())
        },
        "examples": dict(examples),
    }
    return result


def write_markdown(report_path: Path, result, tokenizer_name, seq_lens):
    with report_path.open("w", encoding="utf-8") as out:
        out.write("# Token Stats v0\n\n")
        out.write(f"- tokenizer: `{tokenizer_name}`\n")
        out.write(f"- seq_lens: `{seq_lens}`\n\n")

        if tokenizer_name == "utf8_byte_fallback":
            out.write("> WARNING: This is byte-level fallback sizing, not real model tokenization.\n\n")

        for split, stats in result.items():
            out.write(f"## {split}\n\n")
            out.write(f"- path: `{stats['path']}`\n")
            out.write(f"- bad: `{stats['bad']}`\n\n")

            all_stats = stats["all"]
            out.write("### Overall\n\n")
            write_stats_block(out, all_stats)

            out.write("### By source\n\n")
            for source, s in stats["by_source"].items():
                out.write(f"#### {source}\n\n")
                write_stats_block(out, s)

            out.write("### By bucket\n\n")
            for bucket, s in stats["by_bucket"].items():
                out.write(f"#### {bucket}\n\n")
                write_stats_block(out, s)

            out.write("### Examples\n\n")
            for source, examples in stats["examples"].items():
                out.write(f"#### {source}\n\n")
                for ex in examples:
                    out.write(f"- chars={ex['chars']}, tokens={ex['tokens']}, preview=`{ex['preview']}`\n")
                out.write("\n")


def write_stats_block(out, s):
    keys = [
        "docs",
        "chars",
        "tokens",
        "avg_chars_per_doc",
        "avg_tokens_per_doc",
        "chars_per_token",
        "token_len_min",
        "token_len_p50",
        "token_len_p90",
        "token_len_p95",
        "token_len_p99",
        "token_len_max",
        "char_len_p50",
        "char_len_p95",
        "char_len_max",
    ]

    for k in keys:
        v = s[k]
        if isinstance(v, float):
            out.write(f"- {k}: {v:.4f}\n")
        else:
            out.write(f"- {k}: {v}\n")

    out.write("- seq_estimates:\n")
    for seq_len, est in s["seq_estimates"].items():
        out.write(
            f"  - seq_len={seq_len}: "
            f"full_sequences={est['full_sequences_if_concatenated']}, "
            f"leftover_tokens={est['leftover_tokens']}\n"
        )
    out.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--val", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--tokenizer", type=str, default="")
    parser.add_argument("--allow-byte-fallback", action="store_true")
    parser.add_argument("--seq-lens", type=str, default="1024,2048,4096")
    parser.add_argument("--max-docs", type=int, default=0)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    args = parser.parse_args()

    seq_lens = [int(x.strip()) for x in args.seq_lens.split(",") if x.strip()]
    tokenizer = build_tokenizer(args)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "train": analyze_file(args.train, tokenizer, seq_lens, max_docs=args.max_docs),
        "val": analyze_file(args.val, tokenizer, seq_lens, max_docs=args.max_docs),
    }

    payload = {
        "tokenizer": tokenizer.name,
        "seq_lens": seq_lens,
        "result": result,
    }

    with args.json_out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    write_markdown(args.report, result, tokenizer.name, seq_lens)

    print(f"Wrote report: {args.report}")
    print(f"Wrote json:   {args.json_out}")
    print(f"Tokenizer:    {tokenizer.name}")

    for split in ["train", "val"]:
        s = result[split]["all"]
        print()
        print(f"== {split} ==")
        print("docs:", s["docs"])
        print("tokens:", s["tokens"])
        print("avg_tokens_per_doc:", round(s["avg_tokens_per_doc"], 2))
        print("token_len_p50:", round(s["token_len_p50"], 2))
        print("token_len_p95:", round(s["token_len_p95"], 2))
        for seq_len in seq_lens:
            est = s["seq_estimates"][str(seq_len)]
            print(f"seq_len={seq_len} full_sequences:", est["full_sequences_if_concatenated"])


if __name__ == "__main__":
    main()
