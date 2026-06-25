# AGENTS.md

## Purpose

Own the bapX/RLCoDAR byte-native runtime, objective harness, and local OpenAI-compatible API surface.

## Ownership

This subtree owns:

- `diffusion.py` — byte tokenizer, byte index, diffusion-style canvas runtime, and local fusion synthesis.
- `objectives.py` — masked byte corruption, recovery scoring, and self-improvement hooks.
- `api.py` — local API and CLI entrypoints around CoDAR.
- `__init__.py` — package exports and package-level metadata.

## Local Contracts

- Keep the CoDAR core byte-native and local-first.
- Do not add Torch, NumPy, hosted model calls, databases, services, or deployment platforms to this subtree unless the root contract and issue scope explicitly justify them.
- Treat `ByteIndex` as the current memory/context substrate.
- Treat `MaskedByteDiffusionObjective` as a non-gradient objective harness, not neural training.
- Keep API behavior OpenAI-compatible where already exposed, but do not hide local runtime limits.
- Runtime credentials must come from environment or caller-provided config, not literal values in source.

## Work Guidance

- Keep runtime paths deterministic enough for repeatable tests.
- Prefer small, explicit pure-Python functions.
- Update `project_map.mmd` with `tools/generate_project_map.py` after changes to entrypoints, routes, classes, or function connections.

## Verification

```bash
python3 rlcodar_hyperagi/diffusion.py
python3 rlcodar_hyperagi/objectives.py
python3 tools/generate_project_map.py --check
```

## Child DOCS-INDEX Index

No child contracts currently exist below this folder.
