# AGENTS.md

## Purpose

Own the internal RLM framework subtree that is present in this repository.

## Ownership

This subtree owns framework code such as core orchestration, clients, environments, datasets, communication types, and execution helpers that are already part of this repository.

## Local Contracts

- Do not instruct agents to clone, install, or treat RLM as an external repository for this project.
- Treat this subtree as an internal base/framework layer below the ByteOmniDiffus project.
- Keep changes small and compatibility-aware.
- Do not mix ByteOmniDiffus-specific behavior into generic RLM base abstractions unless a contract and issue explicitly justify it.
- Prefer adapting integration at "rlcodar_hyperagi/" before modifying internal RLM framework internals.
- Do not add hosted provider defaults, GPT-4o defaults, or provider-specific project identity here.

## Work Guidance

- Preserve existing public interfaces unless the issue scope calls for a breaking change.
- Keep environment and client abstractions agent-neutral and vendor-neutral where possible.
- If provider-specific compatibility code exists, describe it as compatibility only, not project identity.
- Update tests and "project_map.mmd" when changing framework entrypoints, clients, environments, or communication flow.

## Verification

uv run pytest
python3 tools/generate_project_map.py --check

## Child DOCS-INDEX Index

No child contracts currently exist below this folder.
