"""Conservative publication checks, not an exhaustive secret/security scanner.

Scan tracked and nonignored source files (or a clean source export). Report only
relative paths and categories, never the matched content. Derived audit tables
and manifests have exactly the same treatment as the original source files.
"""

import re
import subprocess
from pathlib import Path

PATTERNS = {
    "personal_path": re.compile(
        r"/(?:home|Users)/[\w.-]+/|/mnt/[a-z]/Users/|\b[A-Z]:[/\\]Users[/\\]",
        re.IGNORECASE,
    ),
    "private_cloud": re.compile(
        r"(?:drive|docs)[.]google[.]com|codex[-]remote[-]attachments", re.IGNORECASE
    ),
    "token": re.compile(
        r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}"
        r"|sk-[A-Za-z0-9_-]{20,}|AKIA[A-Z0-9]{16})\b"
    ),
    "credential_assignment": re.compile(
        r"(?i)[\"']?(?:api[_-]?key|access[_-]?token|client[_-]?secret|password)[\"']?"
        r"\s*[:=]\s*[\"'][A-Za-z0-9_./+=-]{20,}[\"']"
    ),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |ENCRYPTED )?PRIVATE KEY-----"),
}
EXCLUDED_SUFFIXES = {".pt", ".pth", ".safetensors", ".bin", ".npy", ".npz", ".zip"}
MAX_FILE_BYTES = 100_000_000


def public_paths(root):
    """Do not omit analysis directories or audit manifests from publication checks."""
    root = Path(root).resolve()
    if (root / ".git").exists():
        output = subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=root
        )
        return sorted({root / p.decode() for p in output.split(b"\0") if p})
    generated_root = {".git", ".pytest_cache", ".ruff_cache", ".venv", "build", "dist"}
    result = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if relative.parts[0] in generated_root:
            continue
        if "__pycache__" in relative.parts or any(p.endswith(".egg-info") for p in relative.parts):
            continue
        if path.is_file() or path.is_symlink():
            result.append(path)
    return sorted(result)


def scan_public_paths(root, paths):
    """Return sanitized findings without following symlinks or reading outside root."""
    root = Path(root).resolve()
    findings = []

    def record(relative, category):
        label = str(relative)
        if any(pattern.search(label) for pattern in PATTERNS.values()):
            label = "[redacted path]"
        findings.append({"path": label, "category": category})

    for path in paths:
        path = Path(path)
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError:
            record("[outside root]", "unsafe_path")
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            record(relative, "unsafe_path")
            continue
        if not path.is_file():
            record(relative, "missing_or_nonregular_file")
            continue
        if path.stat().st_size > MAX_FILE_BYTES:
            record(relative, "large_file")
            continue
        if path.suffix.lower() in EXCLUDED_SUFFIXES:
            record(relative, "excluded_payload")
        if path.name.lower() == ".env" or path.name.lower().startswith(".env."):
            record(relative, "environment_file")
        # Scan binary metadata too; an image extension must not bypass a check.
        content = path.read_bytes().decode("utf-8", errors="replace")
        for category, pattern in PATTERNS.items():
            if pattern.search(content) or pattern.search(relative):
                record(relative, category)
    return findings


def verify_public_hygiene(root):
    paths = public_paths(root)
    findings = scan_public_paths(root, paths)
    return {
        "passed": not findings,
        "scope": "heuristic public source hygiene; not a security certification",
        "checked_files": len(paths),
        "findings": findings,
    }
