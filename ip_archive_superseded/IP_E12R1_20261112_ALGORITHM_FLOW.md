# InteractionPrism E12R1 algorithm (exploratory)

```mermaid
flowchart TD
  A[400 E8 heldout instructions] --> B[E8 frozen archive: label-blind inputs, original predictions]
  B --> C[Token-level spans -> word-level players, original offsets]
  C --> D[Calculate v(empty) and all singleton coalitions]
  D --> E[Robust singleton outlier seeds]
  D --> F[All contiguous pairs and triples, exact I2/I3]
  E --> G[Sparse top-node pair interaction graph]
  F --> G
  F --> H[48 clean-cal prompts: family-specific maximum null reference]
  G --> H
  H --> I[Rank size 1/2/3 coalitions without heldout labels]
  I --> J[Select up to four diverse finalists]
  J --> K[Independent clean-context group insertions + controls]
  J --> L[Original group removals and neutral replacements]
  J --> M[All proper subset insertions for multiword candidates]
  K --> N[Norm, coherence, alignment, subset residual]
  L --> N
  M --> N
  N --> O[400-row CSV, 200-poison CSV, per-coalition CSV and JSONL]
  O --> P[Offline labels added ONLY for diagnostics; no new F1 or threshold]
```

## Fixed E8 value game

`v(S)=1-cos(h_M(x_S+Answer)-hbar, h_M(x+Answer)-hbar)`.

`I(S) = Σ_(T ⊆ S) (-1)^(|S|-|T|) v(T)`.

For `|S|=3`, computing only adjacent pairs is insufficient: the skip pair `(i,i+2)` is also explicitly measured.

A strong `G(S)=v(S)-v(empty)` can be carried by an already-strong singleton; do not conflate total coalition effect with `I3` or with target behavior.
