# Proxy-CV and RealMLP iteration

## Motivation

Kaggle discussion 733271, comment 3509899 reports that a sorted 800k/remaining split gives roughly a 0.01 CV–LB gap. The public code sorts by `sample_id`, which is almost monotonic with month.

We use a stricter, leakage-safe variant that does not split month 45:

- train: months 0–44, 797,486 samples;
- validation: months 45–70, 460,151 samples.

Feature ranking, scaling and all preprocessing use training rows only.

## Proxy baseline and feature count

Combined old + Micro-v3 LGB:

- global cosine: **0.13664**;
- monthly mean: 0.13617;
- worst month: 0.11666.

Top-N LGB ablation:

| Features | Proxy cosine |
|---:|---:|
| 64 | 0.13206 |
| 96 | 0.13554 |
| 128 | **0.13701** |
| 160 | 0.13664 |
| 224 | 0.13651 |
| 320 | 0.13696 |
| 434 | 0.13646 |

Top128 was selected for RealMLP. Top320 RealMLP was later rejected: best proxy 0.13005 versus 0.13277 for Top128, with twice the training cost.

## RealMLP v4

Configuration:

- Top128 train-only-ranked features;
- 8-member batch ensemble;
- PBLD feature embeddings;
- widths 384/256/64;
- batch size **1024**;
- **16 epochs**;
- SmoothL1/MSE-dominant objective plus member-level and ensemble-level cosine auxiliaries;
- checkpoint saving every epoch and EMA;
- no validation labels used in preprocessing.

The initial cosine-dominant version peaked at epoch2 then overfit sharply. The MSE-dominant v4 remained stable for all 16 epochs.

## Strict validation

Using a fixed checkpoint recipe (epochs 6/9/11) and a fixed unit blend of enhanced LGB 60% + RealMLP 40%:

| Fold | Enhanced LGB | LGB60 + RealMLP40 | Baseline worst month | Blend worst month |
|---|---:|---:|---:|---:|
| Proxy 45–70 | 0.13664 | about **0.14025** | 0.11666 | 0.11694–0.11961 depending fixed checkpoint |
| Middle 51–60 | 0.13701 | **0.13960** | 0.12285 | **0.12654** |
| Late 62–70 | 0.14638 | **0.14894** | 0.11612 | **0.12439** |

The RealMLP signal therefore improves proxy, middle, late and worst-month stability. It is not a late-only gain.

## Full-data model and candidate

Full-data training completed for 16 epochs at batch1024. Test checkpoint ensemble uses epochs 6/9/11, with member and checkpoint predictions centered and unit-normalized.

Test correlations:

- enhanced LGB vs RealMLP: 0.846;
- current Public-0.137 candidate vs RealMLP: 0.794;
- final 5% conservative candidate vs current candidate: 0.99953.

Exact late reconstruction showed that adding 5% RealMLP to the current Public candidate increased 0.16969 to 0.16977; weights of 10% or more reduced late score. Therefore the conservative file is:

`output/candidate_proxycv_realmlp5.csv`

- 95% current Public-0.137 candidate;
- 5% RealMLP v4 checkpoint ensemble;
- 647,896 aligned unique IDs;
- no NaN or infinity;
- SHA256: `4fe7adf22fc34d037f6043e87719b73c866834d67505971e2c6315cb44139cef`.

This file has not been submitted. Explicit user approval remains mandatory.
