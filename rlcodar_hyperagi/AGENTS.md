# AGENTS.md

## Purpose

Own the ByteOmniDiffus byte-native runtime package, objective harness, and local API/CLI adapter files in this repository.

## Ownership

This subtree owns:

- "diffusion.py" — byte tokenizer, byte index, block-diffusion canvas runtime, local fusion synthesis, and current legacy class/function names.
- "objectives.py" — masked byte corruption, recovery scoring, and self-improvement hooks.
- "api.py" — local API/CLI adapter around the ByteOmniDiffus runtime.
- "__init__.py" — package exports and package-level metadata.

## Local Contracts

- Use ByteOmniDiffus as the project-facing runtime name.
- Do not introduce GPT-4o, hosted provider identity, external provider assumptions, or vendor-specific project language into this subtree.
- Do not describe this package as an external RLM installation target.
- Keep the runtime byte-native and local-first unless a nearer contract and issue justify a change.
- Treat "ByteIndex" as the current memory/context substrate.
- Treat "MaskedByteDiffusionObjective" as a non-gradient objective harness, not neural training.
- Existing code symbols that still contain older names are implementation details until a deliberate compatibility or rename PR changes them.
- Runtime credentials must come from environment or caller-provided config, not literal values in source.

## Work Guidance

- Keep runtime paths deterministic enough for repeatable tests.
- Prefer small, explicit pure-Python functions.
- When using diffusion-language references, describe them as technical mechanics: block diffusion, byte canvas, denoising, low-entropy acceptance, re-noising, self-conditioning, and local fusion.
- Update "project_map.mmd" with "tools/generate_project_map.py" after changes to entrypoints, routes, classes, function connections, or terminology.

## Verification

python3 rlcodar_hyperagi/diffusion.py
python3 rlcodar_hyperagi/objectives.py
python3 tools/generate_project_map.py --check

## Child DOCS-INDEX Index

No child contracts currently exist below this folder.
