# Source Evidence

## Contents

1. Evidence hierarchy
2. Version discipline
3. Call-chain reconstruction
4. Source placement in prose
5. External-paper mapping
6. Validation

## Evidence hierarchy

Prefer evidence in this order:

1. source at the selected immutable revision;
2. tests at the same revision;
3. official documentation for that release;
4. design documents and issues;
5. primary papers or upstream library documentation;
6. secondary explanations only for context.

Treat comments and names as clues, not proof. Confirm behavior through data flow, callers, consumers, or tests.

## Version discipline

Record the revision before analysis. For Git repositories:

```bash
git show <revision>:<path>
git grep <pattern> <revision> -- <path>
git ls-tree -r --name-only <revision>
```

Do not use working-tree line numbers for tagged links. Link to the immutable tag or commit and verify the exact range after the manuscript is written.

## Call-chain reconstruction

For an important abstraction, collect:

```text
definition
constructor
producer
owner
public consumer
state mutation
serialization or process handoff
physical effect
cleanup path
```

Stop when the next calls are ordinary plumbing and no longer change the mental model. A good course does not reproduce every wrapper.

For “what does this function do,” identify:

- input contract;
- output contract;
- state read and written;
- caller expectation;
- downstream consumer;
- why the function exists as a separate boundary;
- failure behavior.

## Source placement in prose

Place each link near the claim it supports. Choose anchors that establish the mechanism; do not enforce a fixed number or link every wrapper.

Good sequence:

```text
definition link
→ explain invariant
→ producer link
→ concrete example
→ consumer link
→ runtime consequence
```

Avoid a final section that merely lists files already needed earlier.

## External-paper mapping

Classify every relationship between a paper and production code:

- **Direct:** implementation explicitly derives from the work.
- **Adapted:** core idea remains, but interfaces or constraints differ.
- **Convergent:** solves the same problem with a related mechanism.
- **Contrast:** useful alternative not implemented here.

State the classification. Compare problem, invariant, mechanism, and operational constraints—not just names.

## Validation

Check four things independently:

1. the path exists at the revision;
2. anchor lines are inside the file;
3. referenced lines contain the claimed definition or behavior;
4. prose does not overstate ownership or causality.

Run `scripts/audit_course.py` for mechanical GitHub-link and Markdown checks. Its success does not replace semantic inspection.

Use `--require-visual` only when the deliverable calls for a visual. It checks for a Mermaid or Markdown image anywhere in each selected document, not at the beginning. It is a syntax-level check: placement, rendering, mathematical correctness, and teaching value still require review. Do not use the flag for text-only lessons.
