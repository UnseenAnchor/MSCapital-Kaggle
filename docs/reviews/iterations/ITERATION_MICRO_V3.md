# Microstructure v3 iteration

## Constraints

- No Kaggle submission without explicit user approval.
- Reject changes that improve only the late fold.
- Normalize every ensemble member to centered unit norm.

## Feature expansion

`src/feat_microstructure_v3.py` streams the raw feather files and creates 342 features. Combined with the original 92 features, the LGB matrix has 434 columns.

New families include:

- level-1/2 mid, spread, depth imbalance, microprice and book slope;
- 10/30/60-second signed trade volume and signed amount;
- correctly signed new/cancel order pressure and side-specific cancellation imbalance;
- short/long-window deltas and cross-stream flow/book interactions.

## Rolling LGB result

Three-seed average, centered cosine:

| fold | old reference | combined 434 | old25 + combined75 |
|---|---:|---:|---:|
| early (41–50) | 0.12747 | 0.13091 | **0.13274** |
| middle (51–60) | 0.13097 | 0.13701 | **0.13815** |
| late (62–70) | 0.13565 | 0.14638 | **0.14706** |

The gain occurs on all folds. It is therefore safer than a late-only tuning result.

## MultiStream multi-seed result

- middle seed/checkpoint ensembles: 0.14024, 0.14048 and 0.14452;
- middle three-seed ensemble: **0.14622**;
- late original seed42 checkpoint ensemble: **0.15661**;
- late seed13 trained with batch1024 was under-converged: 0.14206;
- adding that weak seed reduced late to 0.15423, so it was rejected.

Batch1024 did not OOM, but mmap/CPU transfer became the bottleneck: roughly 8.8 minutes per epoch versus about 3.3 minutes at batch512. Subsequent v2 training should use batch512.

## OOF blend analysis

For common middle/late folds, robust grid search selected a stable neighborhood around:

- legacy LGB: 10%
- microstructure v3 LGB: 30%
- MultiStream v2: 60%

Scores: middle **0.15131**, late **0.16375**. Relative to legacy-LGB10 + v2-90, this improves middle by 0.00304 and late by 0.00468.

Because the current late validation still overstates Public performance, the upload candidate is deliberately more conservative: retain 90% of the Public-0.136 ensemble and add only 10% microstructure-v3 LGB.

## Candidate — not submitted

`output/candidate_micro_v3_conservative.csv`

- rows: 647,896
- finite predictions: yes
- unique/aligned sample IDs: yes
- centered mean: approximately zero
- SHA256: `81af7899cb2da01d8024410b7e7ac83ddeccdef4ddda6ec8da979ec28f9a5313`

Exact composition:

- 90% `candidate_top10_multiscale_conservative.csv`
- 10% `submission_micro_lgb_full_unit.csv`

The reconstructed late score rises from 0.16935 to **0.16969**. The candidate must not be submitted until the user explicitly approves it.
