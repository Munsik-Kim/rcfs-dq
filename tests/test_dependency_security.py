"""Project install/build gates and separately labelled upstream regressions.

Existing upstream checks remain active: no skip or xfail hides their result.
Related native-crash payloads are private audit inputs, not public test fixtures.
"""

import importlib.metadata
import importlib.util
import subprocess
import sys
import tarfile
import tomllib
import unicodedata
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.project_security_gate
def test_all_installation_contracts_exclude_flagged_versions():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    runtime = {r.name: r for r in map(Requirement, project["project"]["dependencies"])}
    build = {r.name: r for r in map(Requirement, project["build-system"]["requires"])}
    for old in ("2.6.0", "2.11.0", "2.12.1", "2.13.0"):
        assert old not in runtime["torch"].specifier
    for old in ("70.2.0", "78.1.0", "82.0.1"):
        assert old not in build["setuptools"].specifier
        assert old not in runtime["setuptools"].specifier
    assert "2.14.0" in runtime["torch"].specifier
    assert "83.0.0" in build["setuptools"].specifier
    assert "83.0.0" in runtime["setuptools"].specifier
    test_requirements = {
        r.name for r in map(Requirement, project["project"]["optional-dependencies"]["test"])
    }
    assert "build" in test_requirements
    pins = {
        r.name: r
        for r in map(
            Requirement,
            [
                line
                for line in (ROOT / "requirements/cpu-constraints.txt").read_text().splitlines()
                if line and not line.startswith("#")
            ],
        )
    }
    assert str(pins["torch"].specifier) == "==2.14.0+cpu"
    assert str(pins["setuptools"].specifier) == "==83.0.0"


@pytest.mark.project_security_gate
def test_installed_dependencies_meet_remediated_contract():
    assert Version("2.14.0") <= Version(importlib.metadata.version("torch")) < Version("2.15")
    assert Version("83.0.0") <= Version(importlib.metadata.version("setuptools")) < Version("84")


@pytest.mark.project_security_gate
def test_removed_package_index_download_boundary():
    # Fixed/current setuptools no longer ships the legacy download implementation.
    assert importlib.util.find_spec("setuptools.package_index") is None


@pytest.mark.parametrize("filename_form,pattern_form", [("NFD", "NFC"), ("NFC", "NFD")])
@pytest.mark.project_security_gate
def test_unicode_exclusion_and_ordinary_file_control(filename_form, pattern_form):
    from setuptools.command.egg_info import FileList

    filename = unicodedata.normalize(filename_form, "excluded_café.txt")
    pattern = unicodedata.normalize(pattern_form, "excluded_café.txt")
    files = FileList()
    files.files = [filename, "ordinary.txt"]
    files.process_template_line("global-exclude " + pattern)
    assert files.files == ["ordinary.txt"]


@pytest.mark.parametrize("filename_form,pattern_form", [("NFD", "NFC"), ("NFC", "NFD")])
@pytest.mark.project_security_gate
def test_supported_sdist_backend_excludes_unicode_file(tmp_path, filename_form, pattern_form):
    (tmp_path / "pyproject.toml").write_text(
        '[build-system]\nrequires=["setuptools>=83,<84"]\n'
        'build-backend="setuptools.build_meta"\n'
        '[project]\nname="packaging-control"\nversion="0.0.0"\n'
        '[tool.setuptools]\npy-modules=["ordinary"]\n'
    )
    (tmp_path / "ordinary.py").write_text("VALUE = 1\n")
    (tmp_path / "ordinary.txt").write_text("public synthetic control\n")
    name = unicodedata.normalize(filename_form, "excluded_café.txt")
    pattern = unicodedata.normalize(pattern_form, "excluded_café.txt")
    (tmp_path / name).write_text("synthetic excluded fixture\n")
    (tmp_path / "MANIFEST.in").write_text("global-include *.txt\nglobal-exclude " + pattern + "\n")
    result = subprocess.run(
        [sys.executable, "-m", "build", "--sdist", "--no-isolation"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    archives = list((tmp_path / "dist").glob("*.tar.gz"))
    assert len(archives) == 1
    with tarfile.open(archives[0]) as archive:
        names = [unicodedata.normalize("NFC", n) for n in archive.getnames()]
    assert any(n.endswith("/ordinary.txt") for n in names)
    assert not any(n.endswith("/excluded_café.txt") for n in names)


@pytest.mark.parametrize("annotation,value", [("list", "[]"), ("tuple", "(1, 2)")])
@pytest.mark.upstream_dependency_sentinel
def test_jit_rejects_untyped_value_annotations_without_process_crash(annotation, value):
    # Isolated process also makes a native crash observable as a failed test.
    source = f"def probe():\n    value: {annotation} = {value}\n    return value\n"
    code = """import sys,torch
try:
    torch.jit.CompilationUnit(sys.argv[1])
except RuntimeError as error:
    if 'without a contained type' not in str(error): raise
    print('REJECTED_WITH_EXCEPTION')
else:
    raise RuntimeError('Untyped value annotation was accepted')
"""
    result = subprocess.run(
        [sys.executable, "-c", code, source], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "REJECTED_WITH_EXCEPTION"


@pytest.mark.upstream_dependency_sentinel
def test_jit_typed_container_control_still_works():
    import torch

    unit = torch.jit.CompilationUnit(
        "def ordinary(values: List[int]) -> int:\n    return len(values)\n"
    )
    assert unit.ordinary([2, 3, 5]) == 3
