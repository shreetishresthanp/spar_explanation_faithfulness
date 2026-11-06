# Ground-truth explanations: claim-by-claim check (training pairs only)

Each row measures one claim on the training decisions (both scenarios of each training pair). **n** is the number of decisions in the subset; **rate** is how often the model did what the claim says. Held-out pairs are excluded. Re-run with `python analysis/mm18_gt_verify.py`.


## GPT-5.2

| Claim | n | Rate |
|---|---|---|
| 1. Passengers vs pedestrians (no red light on the pedestrians): sacrifices the passengers | 567 | 97% |
| 1. ...still does when the pedestrians cross on red (claim: about 3 in 4) | 195 | 74% |
| 1. Exception: passengers outnumber the pedestrians, so it spares the larger group | 31 | 71% |
| 2. Two pedestrian groups, one crossing legally: spares the lawful group (claim: over 90%) | 370 | 93% |
| 3. Two pedestrian groups, no light, equal numbers: stays in lane (claim: over 90%) | 301 | 95% |
| 4. Numbers differ: chooses fewer deaths | 151 | 94% |
| 4. Humans vs animals: spares the humans | 74 | 100% |
| 5. Answer format makes no difference: Case 1 rate, answer first | 616 | 68% |
| 5. ...Case 1 rate, answer last | 864 | 71% |

Claims resting on the rule's coefficients (standardised log-odds of choosing Case 1; ~0 means no effect):

| Factor | Coef | 95% CI |
|---|---|---|
| Case 1 kills the lower-status group | +0.17 | [-0.08, +0.43] |
| Case 1 kills the women | -0.14 | [-0.40, +0.03] |
| Case 1 kills the younger group | -0.55 | [-0.92, -0.20] |
| Case 1 kills the less fit group | +0.02 | [-0.16, +0.23] |
| [ANSWER] asked first | -0.24 | [-0.51, -0.01] |

## Gemini 3 Flash

| Claim | n | Rate |
|---|---|---|
| 1. Two pedestrian groups, one crossing legally: spares the lawful group (claim: about 95%) | 448 | 95% |
| 1. Pedestrians on red vs passengers: hits the pedestrians (claim: about 80%) | 237 | 79% |
| 2. Passengers vs pedestrians, numbers differ: spares the larger group (claim: almost always) | 88 | 99% |
| 2. Humans vs animals: spares the humans | 92 | 100% |
| 3. Passengers vs pedestrians, no red light, equal numbers: sacrifices the passengers (claim: about 80%) | 601 | 83% |
| 3. Two pedestrian groups, no light, equal numbers: stays in lane (claim: about 80%) | 396 | 80% |
| 5. Answer format makes no difference: Case 1 rate, answer first | 872 | 57% |
| 5. ...Case 1 rate, answer last | 924 | 57% |

Claims resting on the rule's coefficients (standardised log-odds of choosing Case 1; ~0 means no effect):

| Factor | Coef | 95% CI |
|---|---|---|
| Case 1 kills the lower-status group | +1.70 | [+1.48, +1.96] |
| Case 1 kills the women | -0.02 | [-0.16, +0.10] |
| Case 1 kills the younger group | -1.48 | [-1.86, -1.24] |
| [ANSWER] asked first | -0.23 | [-0.44, -0.01] |

## Gemma 3 27B

| Claim | n | Rate |
|---|---|---|
| Answer requested after reasoning: picks Case 1 (claim: about 95%) | 612 | 95% |
| Answer requested first: Case 1 rate (for contrast) | 522 | 57% |
| 1. [answer first] Two pedestrian groups, one crossing legally: spares the lawful group (claim: about 90%) | 138 | 89% |
| 2. [answer first] Passengers vs pedestrians: sacrifices the passengers (claim: about 80%) | 242 | 81% |
| 2. [answer first] ...when the pedestrians cross on red | 59 | 81% |
| 2. [answer first] Exception: passengers outnumber pedestrians, so it spares the larger group | 21 | 81% |
| 3. [answer first] Gender scenarios: spares the women (claim: almost always) | 264 | 74% |
| 4. [answer first] Numbers differ: chooses fewer deaths | 69 | 91% |
| 4. [answer first] Humans vs animals: spares the humans | 37 | 95% |

Claims resting on the rule's coefficients (standardised log-odds of choosing Case 1; ~0 means no effect):

| Factor | Coef | 95% CI |
|---|---|---|
| Case 1 kills the women | -1.23 | [-1.60, -1.00] |
| [ANSWER] asked first | -2.30 | [-2.83, -1.98] |
