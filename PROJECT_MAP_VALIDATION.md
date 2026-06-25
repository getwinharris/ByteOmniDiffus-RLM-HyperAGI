# PROJECT_MAP_VALIDATION.md

## Result

Current `project_map.mmd` validation status: **FAIL / STALE**.

The file is useful as a manually written orientation map, but it does not currently satisfy the root DOCS-INDEX contract for a generated repository knowledge graph.

## Evidence

The root `AGENTS.md` contract says:

- `project_map.mmd` is the repository's generated coding knowledge map.
- `project_map.mmd` is not manually maintained.
- Do not write the project map by hand.
- Generate and verify it with `tools/generate_project_map.py`.

The current `tools/generate_project_map.py` produces deterministic hashed Mermaid node IDs through `node_id(...)` and emits generated sections from the actual filesystem scan.

The current `project_map.mmd` does not match that generator output shape. It uses hand-named nodes such as `Repo`, `Identity`, `DRoot`, `ARoot`, `README`, `ProjectMap`, and manually curated subsystem blocks.

Therefore `python3 tools/generate_project_map.py --check` should be expected to fail until the map is regenerated from the checked-out repository.

## Coverage Gaps Found

The current map only lists a small curated subset of top-level folders:

- `rlcodar_hyperagi/`
- `rlm/`
- `tools/`

But the repository visible on `main` includes additional durable folders and files that are not represented in the current generated-map draft, including at least:

- `.github/workflows/`
- `bapxdi_cot/`
- `bapxdi_docs/`
- `examples/`
- `hyperagents/`
- `media/`
- `tests/`
- `visualizer/`
- root docs such as `DIFFUSION_COMPARISON.md`, `KNOWLEDGE_FLOW_COMPARISON.md`, `MERCURY2_ANALYSIS.md`, `RACODAR_IMPLEMENTATION_PLAN.md`, and `WORKING_CODE_COMPARISON.md`

A valid generated map must scan these recursively and represent them according to file type, ownership, and connection status.

## Generator Issues

The generator is directionally better than the prior shallow map, but it still needs testing in the actual checkout.

Known requirements before accepting it:

1. Run `python3 tools/generate_project_map.py` in a clean checkout.
2. Confirm `project_map.mmd` is overwritten by generated hashed-node output.
3. Run `python3 tools/generate_project_map.py --check`.
4. Inspect whether the generated Mermaid file is too large or still readable.
5. Confirm that it indexes all durable folders and does not invent project identity.

## Terminology Validation

The corrected map uses ByteOmniDiffus as the project identity.

It still mentions `CoDARDiffusion` only as a current legacy implementation symbol. This is acceptable only as a temporary compatibility note. A future rename PR should decide whether to rename code symbols and compatibility exports.

## Required Fix

Do not manually patch `project_map.mmd` again.

Next corrective step:

```bash
python3 tools/generate_project_map.py
python3 tools/generate_project_map.py --check
```

Then commit the generated `project_map.mmd` output.

## Status

This validation report exists so the current PR does not falsely claim the project map is fully validated.
