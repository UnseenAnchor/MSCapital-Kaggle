# Domain-shift audit and public LB0.142 anchor

## Train/test adversarial validation

A balanced LightGBM classifier was trained to distinguish train from test using the 434 engineered features. `sample_id`, `month` and `target` were excluded.

- held-out domain AUC: **0.7951**;
- train score median: 0.364;
- test score median: 0.657;
- dominant drift features: volume level, spread, depth, market count and order gap features.

This confirms strong covariate shift.

## OOF by test similarity

On Proxy months45–70, cosine by adversarial-score quintile:

| Quintile | LGB | base v3 | joint v3 | base40+joint60 |
|---:|---:|---:|---:|---:|
| least test-like | 0.1414 | 0.1498 | 0.1497 | 0.1536 |
| 2 | 0.1358 | 0.1475 | 0.1514 | 0.1541 |
| 3 | 0.1421 | 0.1577 | 0.1540 | 0.1602 |
| 4 | 0.1348 | 0.1412 | 0.1480 | 0.1497 |
| most test-like | 0.1274 | 0.1402 | 0.1437 | 0.1469 |

The joint model improves even extreme test-like tails, so its unchanged Public score cannot be localized to a simple adversarial-score subgroup.

## Rejected domain adaptations

### Density-ratio sampling

Weights used `sqrt(p/(1-p))`, clipped and normalized. Effective sample size remained 76.9%, but fixed Proxy ensemble fell to 0.1254 and the most test-like quintile fell to 0.1155. Covariate similarity does not guarantee invariant target conditionals. Rejected before Middle/Late.

### Instance-normalized joint stream

The fourth stream used masked per-sample temporal normalization while base streams retained global levels. Proxy blend reached 0.1535, but the most test-like quintile was 0.1447 versus 0.1469 for the normal joint model. Rejected.

### Sample-wise domain gate

Pre-registered `joint_weight = 0.4 + 0.5 * domain_score` slightly reduced global Proxy/Middle/Late and reduced the most test-like group. Rejected.

## Public LB0.142 slim-pack audit

Source: public Kaggle dataset `yangq369/kaggle-lb0142-upload`, downloaded under `research/lb0142/` and excluded from Git.

The pack contains:

- model definitions and train/infer code;
- five MultiStream member checkpoints and predictions;
- one RealMLP checkpoint and prediction;
- manifest with reported Public LB 0.142;
- frozen reference submission.

Declared formula:

```text
ens5 = mean(unit(v9_big, v9_ctrl, v9_deep, v9_v3grid, v9_v3grid_big))
reference = 0.6 * ens5 + 0.4 * unit(v10)
```

Independent reconstruction from the six member CSV files matches `submission_ref_lb0142.csv` with maximum absolute difference `1.1e-16`. IDs, row count and finite values all pass validation. This is a public competition resource, not private data or leaked labels.

## Fusion candidate

Our chosen anchor is the actually verified Public-0.140 file `candidate_v3_eff1024_conservative20.csv`, not later candidates whose displayed Public remained tied.

- correlation with public LB0.142 reference: **0.90109**;
- formula: 40% ours + 60% public reference after centered unit normalization;
- endpoint-based expected Public center: about **0.14468**;
- accounting for each endpoint's three-decimal rounding, expected range: **0.14416–0.14519**;
- estimated current rank range: around 26–30.

File:

`output/candidate_ours40_public142_60.csv`

Checks:

- 647,896 rows;
- aligned unique IDs;
- no NaN or infinity;
- SHA256: `cb02612534f301930d49c46faa4dba4882fc78af96135a9f74fa6ba71d799c45`.

The candidate has not been submitted. It must be described transparently as a blend with a public Kaggle prediction, and explicit user approval remains mandatory.
