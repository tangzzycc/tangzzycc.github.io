---
name: explain-vllm
description: Explain vLLM architecture, data structures, request lifecycles, scheduling, KV cache, model execution, and GPU operators from the checked-out source code. Use when the user asks how a vLLM concept works, where it is implemented, who calls a function, or how modules and processes interact.
---

# Explain vLLM

Explain the requested concept from purpose to implementation. Treat the checked-out
repository as authoritative; do not substitute a generic vLLM architecture story for
the current code.

## Workflow

1. State the problem the component solves in one or two sentences.
2. Place it in the end-to-end request path.
3. Identify its main data structures and ownership relationships.
4. Trace the real caller-to-callee path with file and function references.
5. Walk through one small concrete example.
6. Separate normal behavior from optional paths such as prefix caching,
   speculative decoding, multimodal input, and distributed execution.

## Explanation Rules

- Answer the exact function, field, or concept named by the user first.
- Start from the nearest meaningful entry point, then move downward through the
  actual control flow.
- Distinguish decision, state, and execution responsibilities. For example:
  Scheduler decides; KVCacheManager manages logical ownership; GPUModelRunner
  prepares tensors; attention backends launch GPU operations.
- Distinguish metadata from physical storage. A KVCacheBlock is metadata; real K/V
  values live in GPU tensors.
- Distinguish block_table from slot_mapping: block_table reads historical KV;
  slot_mapping locates writes for newly computed KV.
- State when a concept is architectural rather than a single class or kernel.
- Avoid claiming every attention backend follows one identical implementation path.

## Output Shape

Prefer this compact structure unless the user requests another format:

1. Definition
2. Architectural position
3. Main execution path
4. Key data structures
5. Concrete example
6. Source locations

Use a small ASCII or Mermaid diagram when relationships are easier to understand
visually. Use clickable source links or local file references when available.
