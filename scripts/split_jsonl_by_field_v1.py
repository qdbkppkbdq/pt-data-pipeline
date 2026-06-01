#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path


def get_nested(obj, key):
    cur = obj
    for part in key.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def safe_name(x):
    return str(x).replace("/", "_").replace(" ", "_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--field", default="bucket")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    inp = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)

    handles = {}
    counts = Counter()

    with inp.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            value = get_nested(obj, args.field)
            if value is None:
                value = "UNKNOWN"
            name = safe_name(value)
            out_path = out_dir / f"{args.prefix}_{name}.jsonl"
            if name not in handles:
                handles[name] = out_path.open("w", encoding="utf-8")
            handles[name].write(json.dumps(obj, ensure_ascii=False) + "\n")
            counts[name] += 1

    for h in handles.values():
        h.close()

    lines = []
    lines.append("# split_jsonl_by_field_v1")
    lines.append("")
    lines.append(f"- input: `{inp}`")
    lines.append(f"- field: `{args.field}`")
    lines.append(f"- out_dir: `{out_dir}`")
    lines.append("")
    lines.append("| value | count | output |")
    lines.append("|---|---:|---|")
    for name, count in counts.most_common():
        lines.append(f"| {name} | {count} | `{out_dir / f'{args.prefix}_{name}.jsonl'}` |")

    report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {len(counts)} splits to {out_dir}")
    print(f"wrote {report}")


if __name__ == "__main__":
    main()
