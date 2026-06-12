# NER Optimization Design - 2026-06-04

## Goal

Improve Chinese medical NER precision/F1 without breaking the existing Streamlit,
RAG, KG, or Graph-RAG workflows.

## Current Signal

The active model has higher recall than precision:

- Precision: 0.5574
- Recall: 0.6293
- F1: 0.5911

This suggests many false-positive entity predictions. More training alone has
plateaued, so the next changes should be optional and measurable.

## Approach

1. Add optional entity confidence thresholding for decoding/evaluation.
   - If a token is predicted as a non-`O` label but its softmax confidence is
     below the threshold, convert it to `O`.
   - Default threshold is `0.0`, preserving existing behavior.

2. Add optional weighted token loss for future training experiments.
   - Default strategy is `none`, preserving existing behavior.
   - `entity` strategy gives all entity labels the same configurable weight.
   - `inverse_sqrt` strategy gives rarer labels moderately higher weights with
     min/max clamps.

## Boundaries

- Do not change `config.yaml` defaults.
- Do not switch active model unless test F1 improves.
- Do not change Streamlit behavior unless the selected model/post-processing
  demonstrably improves.
- Do not add new third-party dependencies.

## Verification

- Unit tests for thresholding and loss-weight calculation.
- Active model evaluation with threshold sweep on dev/test.
- Weighted training only if the decoding change does not produce enough gain.
