"""CPU-only P1 scalar reanalysis of preserved evidence."""

import argparse
import json
from pathlib import Path

from rcfs_dq.pairing_audit import export_audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, help="A new or empty separate output directory")
    parser.add_argument(
        "--check", action="store_true", help="Read-only exact table/hash comparison"
    )
    args = parser.parse_args()
    result = export_audit(
        args.root, args.output or args.root / "analysis/pairing_ablation", check=args.check
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
