# ByteOmniDiffus-RLM-HyperAGI

Status: scratch rebuild.

The old byte-search / CoDAR / RLCoDAR prototype is no longer the active product direction.

## Current target

Build a lean diffusion-style research system around combined remote model-weight records.

A model record groups the remote artifacts that belong together as one dense model object:

- GGUF / ONNX / safetensors weights
- Q8 quantization metadata
- config files
- tokenizer files
- model cards
- dataset links
- source URLs
- capability tags

The system is not a next-token prediction pipeline. It is intended to route and refine over combined model-state records in a diffusion-like way.

## Required first rebuild pieces

- SearXNG search integration
- direct URL indexer
- Hugging Face model and dataset artifact indexer
- combined model-record schema
- Q8 quantized weight metadata scanner
- capability router
- generated `project_map.mmd`

## Rules

- Keep the repository small.
- Do not keep stale prototype code as the product identity.
- Do not add hosted-provider identity.
- Do not describe RLM as an external install target.
- Generate `project_map.mmd` from repository truth, not by hand.

## Active issue

Cleanup and scratch rebuild are tracked in issue #10.
