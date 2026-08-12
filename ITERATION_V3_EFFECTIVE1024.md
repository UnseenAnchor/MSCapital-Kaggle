# v3 MultiStream effective-batch-1024 iteration

## Constraints

- No Kaggle submission without explicit user approval.
- Requested batch size is at least 1024.
- The 400/120 grid cannot fit a physical batch of 1024 on RTX 4070 12GB, so training uses micro-batch 128 with 8-step gradient accumulation: **effective batch 1024**.
- Every run uses 12 epochs. Checkpoints are selected by a fixed cross-fold recipe, not test information.

## RealMLP multi-seed audit

Seeds 13 and 77 were trained for 16 epochs at batch1024 on proxy CV.

- seed13 fixed checkpoint ensemble: proxy 0.13276;
- seed42: 0.13038;
- seed77: 0.12945;
- seed42+13 slightly improved middle from 0.13960 to 0.13970 after LGB blending;
- the same combination reduced late from 0.14894 to 0.14812.

Seed13 and seed77 were rejected. No full-data model or upload candidate was produced from them.

## Gradient accumulation implementation

`src/train_multistream_grid.py` now supports:

- `ACCUM_STEPS`;
- correct `loss / ACCUM_STEPS` scaling;
- optimizer steps only after accumulation;
- correction for a final partial accumulation;
- resumable model/optimizer/scaler state;
- validation-free full-data training;
- logging of micro, accumulation and effective batch sizes.

## Model

- grid: v3, Market 400 / Transaction 120 / Order 120;
- CNN–Transformer MultiStream with cross-stream encoder;
- `d_model=64`, two layers;
- learning rate 0.0006;
- loss: 70% centered cosine + 30% SmoothL1;
- micro-batch 128 × accumulation 8;
- effective batch 1024;
- 12 epochs;
- fixed checkpoint ensemble: epochs 4/5/6.

## Validation

### Proxy: months 0–44 → 45–70

- enhanced LGB: 0.13664;
- v3 checkpoint ensemble 4/5/6: **0.14752**;
- LGB40 + v3 60: **0.15419**;
- monthly mean: 0.15114;
- worst month: **0.13426**, versus LGB 0.11666;
- LGB/v3 correlation: 0.713.

### Middle: months 0–50 → 51–60

- enhanced LGB: 0.13701;
- v3 checkpoint ensemble: **0.14825**;
- LGB40 + v3 60: **0.15299**;
- worst month: **0.13576**, versus LGB 0.12285;
- LGB/v3 correlation: 0.756.

### Late: months 0–61 → 62–70

- enhanced LGB: 0.14638;
- v3 checkpoint ensemble 4/5/6: **0.15995**;
- LGB40 + v3 60: **0.16575**;
- alternate 5/6/8 ensemble reaches 0.16741 after blending, but was not selected because 4/5/6 is the fixed cross-fold recipe;
- worst month: **0.13835**, versus LGB 0.11612;
- LGB/v3 correlation: 0.727.

The model improves global cosine, monthly mean and worst month on all three validation schemes.

## Full model and candidate

Full-data training completed for all 12 epochs. Test inference uses checkpoints 4/5/6, with each checkpoint centered and unit-normalized before averaging.

Test correlations:

- new v3 vs current Public-0.138 candidate: 0.8845;
- final candidate vs Public-0.138 candidate: 0.9955.

Candidate:

`output/candidate_v3_eff1024_conservative20.csv`

Composition:

- 80% current Public-0.138 candidate;
- 20% new v3 effective-batch-1024 ensemble.

Validation:

- 647,896 rows;
- aligned unique sample IDs;
- no NaN or infinity;
- SHA256: `56ae874b8332eff0f1fcda37dd547551e46570402b6fdbcb6b5f887c83958194`.

The candidate has not been submitted. Explicit user approval remains mandatory.
