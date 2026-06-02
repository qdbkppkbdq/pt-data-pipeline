#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.sources import SourceRegistryError, validate_source_registry  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate source registry v1 YAML files.")
    parser.add_argument("--config-dir", default="configs/sources")
    parser.add_argument("--check-paths", action="store_true")
    args = parser.parse_args()

    try:
        summary = validate_source_registry(args.config_dir, check_paths=args.check_paths)
    except SourceRegistryError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False, indent=2))
        raise SystemExit(1) from exc

    summary["ok"] = True
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
