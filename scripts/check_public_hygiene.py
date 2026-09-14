"""Read-only publication hygiene gate; no matched secret content is printed."""

import argparse
import json
from pathlib import Path

from rcfs_dq.public_hygiene import verify_public_hygiene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    report = verify_public_hygiene(args.root)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
