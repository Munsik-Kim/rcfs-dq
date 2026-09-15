"""Read-only public closeout verification; works without private raw or models."""

import argparse
import json
from pathlib import Path

from rcfs_dq.closeout_verification import verify_closeout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        report = verify_closeout(args.root)
    except (ValueError, KeyError, OSError, TypeError):
        print(json.dumps({"passed": False, "error": "Closeout scalar verification failed"}))
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
