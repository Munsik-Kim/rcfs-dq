# Security Policy

## Supported Versions

Security fixes are applied to the current `main` branch and the latest public release, if any.

## Reporting a Vulnerability

Please do not report suspected security vulnerabilities in public GitHub issues.
Use GitHub's private vulnerability reporting feature for this repository when available.

Include:

- affected commit or version,
- reproduction steps,
- expected and observed behavior,
- potential impact.

Do not include real credentials, private keys, proprietary model weights, or sensitive data in reports.

Research-result disagreements, benchmark discrepancies, or scientific-reproducibility questions
that do not create a software-security risk should be reported through normal issues instead.

## Supported Input Boundary

RCFS-DQ does not accept untrusted TorchScript source and does not use `torch.jit`
as part of its supported public execution path. Models are supplied as
already-constructed objects to the adapter. Direct use of PyTorch JIT APIs is
outside the RCFS-DQ supported input boundary.

Models, schedulers and continuation/update callbacks are trusted caller-supplied
Python objects. The adapter is not a sandbox or a strict `nn.Module` type validator.
A supplied object can execute its own code; passing an object does not make it
trusted. Public audit scripts parse scalar JSON/CSV data and paths, not executable
model recipes, checkpoint payloads or source-code plugins.

Do not execute untrusted model code, TorchScript source, checkpoints,
pickle/Python artifacts, or other executable model payloads solely because they
are compatible with the installed PyTorch version.

## Dependency Sentinels and Reporting Scope

An upstream native-crash sentinel may remain for direct PyTorch JIT stress inputs.
This is tracked separately and is not evidence that arbitrary JIT input is safely
handled. Official advisory behavior, related upstream stress observations and
supported RCFS entry points must be assessed separately. Test-only JIT probes are
not product input endpoints; no runtime monkeypatch suppresses their behavior.

A supported RCFS input path to unsafe compilation/deserialization is reportable
and blocks publication; this boundary is not a blanket exclusion of dependency
bugs. Changes to loaders, plugins, callbacks or configuration interpretation
require a fresh reachability review. This policy is not a security certification.
