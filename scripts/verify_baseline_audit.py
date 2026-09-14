"""Independently verify the public retrospective baseline audit, CPU scalar only."""

import argparse
import csv
import io
import json
import math
from pathlib import Path

from rcfs_dq.baseline_independent import verify_baseline_audit


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--audit", type=Path)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Read-only verification, including the saved verification report",
    )
    parser.add_argument(
        "--pairing", action="store_true", help="Verify the separate P1 scalar audit"
    )
    args = parser.parse_args()
    audit = args.audit or args.root / (
        "analysis/pairing_ablation" if args.pairing else "analysis/baseline_audit"
    )
    expected = args.root / (
        "analysis/pairing_ablation" if args.pairing else "analysis/baseline_audit"
    )
    if any(path.is_symlink() for path in (audit, *audit.parents)) or not audit.is_dir():
        parser.error("Audit must be an existing nonsymlink directory")
    if (
        not args.check
        and audit.resolve().is_relative_to(args.root.resolve())
        and audit.resolve() != expected.resolve()
    ):
        parser.error("Refusing to write verification outputs inside another source directory")
    for name in ("independent_verification.json", "main_vs_independent.csv"):
        target = audit / name
        if target.is_symlink() or (target.exists() and not target.is_file()):
            parser.error("Refusing a non-regular verification output")
        if not args.check and audit.resolve() != expected.resolve() and target.exists():
            parser.error("External verification outputs must be new; use --check for readback")
    if args.pairing:
        from rcfs_dq.pairing_independent import verify_pairing_audit

        report, rows = verify_pairing_audit(args.root, audit)
    else:
        report, rows = verify_baseline_audit(args.root, audit)
    report = _json_safe(report)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    comparison_payload = stream.getvalue().encode("utf-8")
    if args.check:
        saved = json.loads((audit / "independent_verification.json").read_text())
        if saved != report:
            print("Saved independent report differs from fresh verification")
            return 1
        if (audit / "main_vs_independent.csv").read_bytes() != comparison_payload:
            print("Saved discrepancy inventory differs from fresh verification")
            return 1
    else:
        previous_report = audit / "independent_verification.json"
        if previous_report.exists():
            previous = json.loads(previous_report.read_text())
            if (
                not isinstance(previous, dict)
                or previous.get("scope") != report["scope"]
                or previous.get("status") not in {"PASS", "FAIL"}
                or previous.get("imports_main_analysis") is not False
                or not isinstance(previous.get("implementation_inventory"), list)
            ):
                parser.error("Unrecognized canonical verification report; preserve and review")
        previous_comparison = audit / "main_vs_independent.csv"
        if previous_comparison.exists() and previous_comparison.read_bytes() != comparison_payload:
            parser.error("Changed comparison data must be preserved and reviewed")
        (audit / "independent_verification.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )
        (audit / "main_vs_independent.csv").write_bytes(comparison_payload)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
