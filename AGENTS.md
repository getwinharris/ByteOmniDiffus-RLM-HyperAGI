# AGENTS.md

## DOCS-INDEX Framework

- DOCS-INDEX is a high-performance `AGENTS.md` hierarchy installed here.
- Agents must follow DOCS-INDEX instructions across all edits.
- `AGENTS.md` files are binding work contracts for their subtrees.
- Work products, source materials, instructions, records, assets, and durable docs must stay understandable from the nearest applicable `AGENTS.md` plus every parent `AGENTS.md` above it.

## Core Contract

Agents must:

- Keep architecture lean.
- Challenge unnecessary layers.
- Consolidate existing plans before creating new ones.
- Prefer durable repository contracts over conversation memory.
- Keep the project understandable from files, docs, maps, and verification records.
- Read project files, contracts, tools, and generated maps before making changes.
- Do not rely on memory when repository contracts exist.
- Re-read the applicable DOCS-INDEX chain in the current session before editing.

## Agent Neutrality

DOCS-INDEX is agent-neutral.

Repository contracts must not depend on a specific AI model, coding agent, IDE, editor, MCP client, or vendor.

Repository knowledge should remain portable across agent systems.

Agents are replaceable. Contracts are durable.

## Project Direction

For this project, act as the project's technical steward.

Prioritize:

- Clear purpose.
- Practical implementation.
- Durable documentation.
- Traceable decisions.
- Verifiable workflows.
- Buildable product outcomes.

Do not add tools, services, frameworks, databases, APIs, deployment paths, or abstractions unless the nearest project contract justifies them.

## Repository Knowledge System

DOCS-INDEX provides the repository contract hierarchy.

`project_map.mmd` provides the generated repository knowledge graph.

Together they form the repository memory system.

- `AGENTS.md` explains rules, ownership, responsibilities, workflows, and operating contracts.
- `project_map.mmd` explains structure, relationships, dependencies, routes, services, tools, implementation status, duplicate patterns, connected functions, and unconnected functions.

Agents should consult both before making significant changes.

## Project Contracts

Every durable project area should define its operating contract in the nearest `AGENTS.md`.

A project contract may record:

- Purpose.
- User or business goal.
- Selected stack.
- Services.
- Environment needs.
- Connected tools.
- Routing.
- Deployment commands.
- Verification steps.
- Current state.
- Non-goals.

## Design Contracts

Every visual project should include a `Design.md`.

Project `AGENTS.md` must route UI, brand, layout, image, and visual work to `Design.md` before implementation.

Do not scaffold visual project code before creating:

- `AGENTS.md`
- `Design.md`

This repository is currently a byte-native research/runtime project, not a visual product. Create `Design.md` only if visual surfaces become durable project scope.

## GitHub Issue, PR, and Versioning Workflow

Treat feature requests, bug reports, production issues, and implementation tasks as GitHub issue work unless the user explicitly says the turn is research-only, planning-only, or no-GitHub.

Before implementation:

- Create or identify a GitHub issue.
- Record the problem statement.
- Define scope.
- Add acceptance criteria.
- Add verification notes.

Branch naming:

- When an issue exists: `agent/issue-<number>-<slug>`
- For explicitly issue-less work: `agent/<type>-<slug>`

Branch names must describe the work, not the implementation tool.

Do not encode specific agent products, vendors, models, IDEs, or runtimes into branch names.

Do not merge directly to `main` for feature or fix work unless the user explicitly authorizes it.

Open a pull request with:

- Linked issue.
- Summary.
- Verification.
- Version impact.

Every PR must declare version impact:

- `none`
- `patch`
- `minor`
- `major`

`VERSION` is the project version source when present.

`CHANGELOG.md` records user-facing, product, architecture, and workflow changes before merge or release.

Use semantic versioning.

For pre-1.0 projects:

- Use `minor` for significant product, architecture, or workflow changes.
- Use `patch` for fixes, docs corrections, and small workflow improvements.

Release tags should use `vX.Y.Z`.

## Project Map

`project_map.mmd` is the repository's generated coding knowledge map.

It exists to make the repository work like a NotebookLM-style source graph for coding agents.

The map should systematically index:

- Files and folders.
- `AGENTS.md` hierarchy.
- Source entrypoints.
- Tool-calling functions.
- Internal functions.
- Routes and handlers.
- UI components.
- Services.
- Scripts.
- Workflows.
- Configuration files.
- Generated files.
- Manual files.
- Connected functions.
- Unconnected functions.
- Duplicate or overlapping functions.
- Missing expected links.
- Stale or intentionally unimplemented areas.

`project_map.mmd` is not a manually maintained knowledge graph.

Rules:

- If `project_map.mmd` already exists, use it.
- If `tools/generate_project_map.py` already exists, use it.
- If `tools/generate_project_map.py` does not exist, create it before relying on generated mapping.
- If `project_map.mmd` does not exist, generate it with `tools/generate_project_map.py`.
- The generator must scan the repository systematically.
- The generator must index enough structure to help agents refine the project.
- The generator must identify connected, unconnected, duplicate, generated, and manual project areas where possible.
- Keep the generator deterministic.
- Update the generated map after changes affecting architecture, routing, tools, capabilities, docs, workflows, or implementation status.
- Verify the generated map before closeout when mapped files or the generator changed.

Recommended commands:

```bash
python3 tools/generate_project_map.py
python3 tools/generate_project_map.py --check
```

After the project map exists, `AGENTS.md` should only reference the generated project map contract. Do not manually duplicate project knowledge that belongs in `project_map.mmd`.

## Read Before Editing

1. Read the root `AGENTS.md`.
2. Identify every file or folder expected to change.
3. Walk from the repository root to each target path.
4. Read every `AGENTS.md` found along each route.
5. If a parent `AGENTS.md` lists a child `AGENTS.md` whose scope contains the path, read that child and continue from there.
6. Use the nearest `AGENTS.md` as the local contract and parent docs for repo-wide rules.
7. If docs conflict, the closer doc controls local work details, but no child doc may weaken DOCS-INDEX.

## Update After Editing

Every meaningful change requires a DOCS-INDEX pass before the task is done.

Update the closest owning `AGENTS.md` when a change affects:

- Purpose, scope, ownership, or responsibilities.
- Durable structure, contracts, workflows, or operating rules.
- Required inputs, outputs, permissions, constraints, side effects, or artifacts.
- User preferences about behavior, communication, process, organization, or quality.
- `AGENTS.md` creation, deletion, move, rename, or index contents.

Update parent docs when parent-level structure, ownership, workflow, or child index changes.

Update child docs when parent changes alter local rules.

Remove stale or contradictory text immediately.

Small edits that do not change behavior or contracts may leave docs unchanged, but the DOCS-INDEX pass still must happen.

## Hierarchy

- Root `AGENTS.md` is the DOCS-INDEX rail: project-wide instructions, global preferences, durable workflow rules, and the top-level Child DOCS-INDEX Index.
- Child `AGENTS.md` files own domain-specific instructions and their own Child DOCS-INDEX Index.
- Each parent explains what its direct children cover and what stays owned by the parent.
- The closer a doc is to the work, the more specific and practical it must be.

## Child Doc Shape

Create a child `AGENTS.md` when a folder becomes a durable boundary with its own:

- Purpose.
- Rules.
- Responsibilities.
- Workflow.
- Materials.
- Quality standards.

Default section order:

```markdown
# AGENTS.md

## Purpose

## Ownership

## Local Contracts

## Work Guidance

## Verification

## Child DOCS-INDEX Index
```

`Work Guidance` must reflect current standards.

If there are no specific standards or instructions yet, leave it empty.

`Verification` must reflect an existing check.

If no verification framework exists yet, leave it empty and update it when one exists.

## Style

- Keep docs concise, current, and operational.
- Document stable contracts, not diary entries.
- Put broad rules in parent docs and concrete details in child docs.
- Prefer direct bullets with explicit names.
- Do not duplicate rules across many files unless each scope needs a local version.
- Delete stale notes instead of explaining history.
- Trim obvious statements, repeated rules, misplaced detail, and warnings for risks that no longer exist.

## Closeout

1. Re-check changed paths against the DOCS-INDEX chain.
2. Update nearest owning docs and any affected parents or children.
3. Refresh every affected Child DOCS-INDEX Index.
4. Remove stale or contradictory text.
5. Create `tools/generate_project_map.py` only when missing and needed.
6. Regenerate `project_map.mmd` when relevant.
7. Run existing verification when relevant.
8. Report any docs intentionally left unchanged and why.

## User Preferences

When the user requests a durable behavior change, record it here or in the relevant child `AGENTS.md`.

Current durable project preference:

- Before discussing or changing this project, verify from repository files first.
- Use repository files as source of truth over chat memory.
- Keep project summaries organized with separate titled sections for Overview, Projects, and Work when reporting back.

## Child DOCS-INDEX Index

Top-level ownership:

- `rlcodar_hyperagi/AGENTS.md` — owns the byte-native CoDAR runtime, objective harness, and OpenAI-compatible local API.
- `rlm/AGENTS.md` — owns the inherited Recursive Language Model base framework and upstream-style environment/client abstractions.
- `tools/AGENTS.md` — owns repository maintenance scripts, especially generated map tooling.
- `project_map.mmd` — generated repository knowledge map. Do not hand-edit.
- `systematicprojectmap.mmd` — earlier manual orientation map. Keep only as a human planning artifact unless replaced by generated map workflow.
- `README.md` — public project overview. Keep aligned with root contract and generated map.
- `pyproject.toml` — packaging, dependencies, and test configuration.

This root contract owns files that do not have a closer child `AGENTS.md`.
