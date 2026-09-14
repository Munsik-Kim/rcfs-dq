"""Reproduce the retrospective scalar baseline audit; never launch a model."""

import argparse
import json
from pathlib import Path

from rcfs_dq.baseline_audit import export_audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--output", type=Path, help="Separate output directory, e.g. an empty temp dir"
    )
    parser.add_argument(
        "--check", action="store_true", help="Read-only exact table/hash comparison"
    )
    args = parser.parse_args()
    destination = args.output or args.root / "analysis/baseline_audit"
    result = export_audit(args.root, destination, check=args.check)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
