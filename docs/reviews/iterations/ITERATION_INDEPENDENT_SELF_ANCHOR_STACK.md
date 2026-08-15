# Independent Self-Anchor Stack

## Objective

After Public ref `55537873` remained 0.144, remove the old Public-derived 80% component and optimize only train-generated predictions.

## Components

- rolling LGB;
- RealMLP;
- original v3 MultiStream;
- Joint v3;
- v2+v3 Multi-Resolution.

No `candidate_proxycv_realmlp5`, no public LB prediction, and no previous submission-derived file was used.

## Fixed robust weights

```text
LGB 20% + RealMLP 15% + v3 15% + Joint 35% + Multi-Resolution 15%
```

## Cross-fold metrics

| Fold | Cosine | Monthly mean | Worst month |
|---|---:|---:|---:|
| Proxy | 0.15865 | 0.15503 | 0.13639 |
| Middle | 0.15603 | 0.15628 | 0.14154 |
| Late | 0.17169 | 0.16152 | 0.14177 |

The stack is stronger than the original v3 in all three fold-level global scores.

## Test candidate

Generated, not submitted:

`output/candidate_independent_stack_public60_40.csv`

Formula:

```text
60% public LB0.142 reference + 40% independent self stack
```

SHA256:

`e7679a75c1118ae7aac35a71abeec639128e9892d61a3b6ee5f6142a183d3595`

Test correlation with current Public0.144 candidate: `0.99444`.

## Decision

- Candidate is retained for a future explicitly approved submission;
- no second upload is made now;
- despite strong cross-fold scores, the high test correlation means the candidate is not yet evidence of a Top10 jump;
- next review must focus on whether this independent stack has enough expected self-anchor strength, not on adjacent weight search.
