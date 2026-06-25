# AGENTS.md

## Purpose

Own the inherited Recursive Language Model base framework.

## Ownership

This subtree owns upstream-style RLM framework code such as core orchestration, clients, environments, datasets, communication types, and execution helpers.

## Local Contracts

- Treat this subtree as framework/base-layer code, not the bapX CoDAR runtime.
- Keep changes small and compatibility-aware.
- Do not mix project-specific RLCoDAR behavior into generic RLM base abstractions unless a contract explicitly says so.
- Prefer adapting integration at `rlcodar_hyperagi/` before modifying inherited RLM internals.

## Work Guidance

- Preserve existing public interfaces unless the issue scope calls for a breaking change.
- Keep environment and client abstractions agent-neutral and vendor-neutral where possible.
- Update tests and `project_map.mmd` when changing framework entrypoints, clients, environments, or communication flow.

## Verification

```bash
uv run pytest
python3 tools/generate_project_map.py --check
```

## Child DOCS-INDEX Index

No child contracts currently exist below this folder.
