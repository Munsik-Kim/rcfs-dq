"""Independently verify P0/P1 scalars and the separate public audit manifest."""

import argparse
import json
from pathlib import Path

from rcfs_dq.public_audits import verify_public_audits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        result = verify_public_audits(args.root)
    except (ValueError, OSError, KeyError, TypeError, AssertionError):
        # Do not emit untrusted file contents, credentials, or absolute paths.
        print(json.dumps({"passed": False, "error": "Public scalar audit verification failed"}))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
