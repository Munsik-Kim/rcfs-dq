"""Index already-derived scalar P0/P1 outputs without modifying original evidence."""

import argparse
import json
from pathlib import Path

from rcfs_dq.public_audits import MANIFEST_NAME, manifest_from_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    result = manifest_from_files(root)
    destination = root / MANIFEST_NAME
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if destination.is_symlink():
        parser.error("Refusing symlink manifest destination")
    if destination.exists():
        if destination.read_text() != payload:
            parser.error("Existing manifest differs; preserve and review before regeneration")
    else:
        destination.write_text(payload, encoding="utf-8")
    print(json.dumps({"artifacts": len(result["artifacts"]), "model_calls": 0}))


if __name__ == "__main__":
    main()
