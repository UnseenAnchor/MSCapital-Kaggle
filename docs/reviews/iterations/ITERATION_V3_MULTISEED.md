# v3 MultiStream second-seed iteration

## Pre-registered gate

Seed13 used exactly the seed42 configuration:

- v3 grid 400/120;
- d_model 64, two Transformer layers;
- micro-batch128 × accumulation8 = effective batch1024;
- learning rate 0.0006;
- 70% cosine + 30% SmoothL1;
- 12 epochs;
- fixed checkpoint ensemble 4/5/6.

The run progressed Proxy → Middle → Late only if global cosine, monthly mean and worst month improved after blending. Seed weights were fixed at seed42 80% + seed13 20% after the Middle gate.

## Proxy gate

v3 single-model ensemble:

- seed42: global 0.14752, monthly mean 0.14376, worst 0.12158;
- seed13: global 0.14503, monthly mean 0.14307, worst 0.12047;
- 60/40 exploratory blend showed the maximum, but 80/20 was selected conservatively before later folds;
- seed42/seed13 correlation: 0.893.

At 40% seed13, global reached 0.15043 and worst month 0.12619. Proxy passed.

## Middle gate at pre-registered 20% seed13

- seed42 global: 0.14825;
- multiseed global: **0.14889**;
- monthly mean: 0.14844 → **0.14904**;
- worst month: 0.13005 → **0.13180**;
- monthly standard deviation decreased;
- seed correlation: 0.914.

Middle passed without changing checkpoint or seed weights.

## Late gate at pre-registered 20% seed13

- seed42 global: 0.15995;
- multiseed global: **0.16182**;
- monthly mean: 0.14987 → **0.15151**;
- worst month: 0.12783 → **0.13003**;
- seed correlation: 0.926.

Late passed. Full-data seed13 was therefore trained for all 12 epochs.

## Candidate

The Public-0.140 candidate already contains 20% seed42 v3. The new candidate changes only that member:

- 80% existing pre-v3 Public-0.138 base;
- 20% v3 multiseed, where v3 = seed42 80% + seed13 20%.

File:

`output/candidate_v3_eff1024_multiseed20.csv`

Checks:

- 647,896 rows;
- unique aligned IDs;
- no NaN or infinity;
- candidate correlation with Public-0.140 candidate: 0.999875;
- SHA256: `ed133a067aa0cab8eb27e8662515ab0802fc3d54b0542aa013823342aa389297`.

Exact late reconstruction:

- Public-0.140 recipe: 0.170865;
- multiseed replacement: **0.171012**;
- delta: +0.000146.

This is a rigorously validated but small change. Because Kaggle reports three decimals, Public may remain displayed as 0.140 or move to 0.141. The file has not been submitted; explicit user approval remains mandatory.
