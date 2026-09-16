"""Supported input-boundary regression; no malformed JIT source is executed."""

import ast
import hashlib
import inspect
import json
import tomllib
from pathlib import Path

import pytest
import torch

from rcfs_dq.dit_adapter import DDIMPath
from rcfs_dq.geometry import finite_horizon_response

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.project_security_gate
SURFACES = sorted(
    [*ROOT.joinpath("src").rglob("*.py"), *ROOT.joinpath("scripts").glob("*.py")]
    + list(ROOT.joinpath("examples").glob("*.py"))
)


def dotted(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return dotted(node.value) + "." + node.attr
    return ""


@pytest.mark.parametrize("path", SURFACES, ids=lambda p: str(p.relative_to(ROOT)))
def test_public_api_has_no_torchscript_source_entrypoint(path):
    tree = ast.parse(path.read_text())
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            aliases.update((a.asname or a.name.split(".")[0], a.name) for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            aliases.update((a.asname or a.name, node.module + "." + a.name) for a in node.names)
    banned = ("torch.jit", "torch._C", "pickle", "cloudpickle", "dill", "importlib")
    assert not any(v == b or v.startswith(b + ".") for v in aliases.values() for b in banned)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = dotted(node.func)
        first, _, suffix = name.partition(".")
        expanded = aliases.get(first, first) + ("." + suffix if suffix else "")
        assert expanded not in {"exec", "eval", "compile", "__import__", "torch.load"}
        assert not any(expanded == b or expanded.startswith(b + ".") for b in banned)
        assert not name.endswith((".from_pretrained", ".load_state_dict"))
        # No dynamic lookup of a compiler/loader via reflection on these surfaces.
        assert name not in {"getattr", "globals", "locals"}


class DDIMScheduler:
    """Harmless constructor protocol fixture, not a diffusion implementation."""

    def set_timesteps(self, steps, device):
        self.timesteps = torch.arange(steps, device=device)


class ConstructedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))

    def forward(self, *args, **kwargs):
        raise AssertionError("Boundary test must not perform model inference")


def test_adapter_accepts_constructed_model_not_source_code():
    model, scheduler = ConstructedModel(), DDIMScheduler()
    path = DDIMPath(model, scheduler, [0], num_steps=2)
    assert path.model is model and path.scheduler is scheduler
    assert list(inspect.signature(DDIMPath).parameters) == [
        "transformer",
        "scheduler",
        "class_ids",
        "num_steps",
    ]
    assert not hasattr(path, "load_model")


@pytest.mark.parametrize("as_path", [False, True])
def test_source_string_or_checkpoint_path_is_not_deserialized(tmp_path, as_path):
    payload = tmp_path / "harmless_model_input.txt" if as_path else "NOT_EXECUTABLE_SOURCE"
    with pytest.raises(AttributeError, match="parameters"):
        DDIMPath(payload, DDIMScheduler(), [0], num_steps=2)
    assert not list(tmp_path.iterdir())


def test_continuation_string_is_not_compiled():
    with pytest.raises(TypeError, match="not callable"):
        finite_horizon_response("NOT_EXECUTABLE_SOURCE", torch.ones(1, 2), torch.ones(1, 2), 0.5)


def test_cli_has_no_dynamic_jit_compilation_path():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert not project.get("scripts") and not project.get("entry-points")
    assert not list(ROOT.joinpath("src").rglob("__main__.py"))
    allowed = {"--root", "--output", "--check", "--audit", "--pairing"}
    observed = set()
    for path in ROOT.joinpath("scripts").glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call) and dotted(node.func).endswith(".add_argument"):
                assert node.args and isinstance(node.args[0], ast.Constant)
                observed.add(node.args[0].value)
    assert observed == allowed


def test_only_product_subprocess_is_static_git_file_inventory():
    paths = []
    for path in SURFACES:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call) and dotted(node.func).startswith("subprocess."):
                paths.append(path.relative_to(ROOT).as_posix())
                assert dotted(node.func) == "subprocess.check_output"
                assert ast.literal_eval(node.args[0]) == [
                    "git",
                    "ls-files",
                    "--cached",
                    "--others",
                    "--exclude-standard",
                    "-z",
                ]
                assert not any(k.arg == "shell" for k in node.keywords)
    assert paths == ["src/rcfs_dq/public_hygiene.py"]


def test_dependency_jit_sentinel_is_not_product_api():
    test_path = ROOT / "tests/test_dependency_security.py"
    tree = ast.parse(test_path.read_text())
    upstream = [
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and any(
            dotted(d) == "pytest.mark.upstream_dependency_sentinel" for d in node.decorator_list
        )
    ]
    assert set(upstream) == {
        "test_jit_rejects_untyped_value_annotations_without_process_crash",
        "test_jit_typed_container_control_still_works",
    }
    assert test_path not in SURFACES
    assert not any(
        isinstance(n, ast.Attribute) and n.attr in {"xfail", "skip", "skipif"}
        for n in ast.walk(tree)
    )


def test_boundary_receipt_binds_all_supported_source_bytes():
    receipt = json.loads((ROOT / "security/JIT_BOUNDARY_REVIEW.json").read_text())
    records = receipt["product_source_bindings"]
    assert {r["path"] for r in records} == {p.relative_to(ROOT).as_posix() for p in SURFACES}
    for record in records:
        assert hashlib.sha256((ROOT / record["path"]).read_bytes()).hexdigest() == record["sha256"]
    assert receipt["reachability"] == "UNREACHABLE_FROM_SUPPORTED_RCFS_API"
    assert receipt["related_same_CVE"] == "NOT_ESTABLISHED"
    assert receipt["security_certification"] is False


def test_public_receipt_keeps_classification_without_native_crash_payload():
    receipt = json.loads((ROOT / "security/JIT_BOUNDARY_REVIEW.json").read_text())
    assert receipt["detailed_reproducers_included"] is False
    assert receipt["official_behavior"] == "OFFICIAL_PATCH_BEHAVIOR_CONFIRMED"
    official = [r for r in receipt["rows"] if r["reproducer_id"] == "OFFICIAL_CVE_REPRODUCER"]
    related = [r for r in receipt["rows"] if r["reproducer_id"].startswith("RELATED_")]
    assert len(official) == 2 and all(r["controlled_exception"] for r in official)
    assert len(related) == 4 and all(r["native_crash"] for r in related)
    for row in receipt["rows"]:
        assert set(row) == {
            "reproducer_id",
            "python",
            "torch",
            "torch_git",
            "exit_code",
            "signal",
            "controlled_exception",
            "native_crash",
            "compiled",
        }
