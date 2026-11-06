# %% [markdown]
# # Verify each claim in the ground-truth explanations against the TRAINING decisions
#
# Run from the repo root after `python analysis/mm18_gt_prep.py`: `python analysis/mm18_gt_verify.py`.
# Writes experiments/moral_machines/mm18_gt_claims.md: one row per claim with the training subset it's measured on,
# how many decisions that is, and how often the model did what the claim says. Held-out pairs are excluded.
# Claims about demographics rest on the rule's coefficients, not raw rates: the generator lays the cases out
# asymmetrically (e.g. with pedestrians only, Case 1 always kills the theme's first group), so raw rates confound them.

# %%
import os
import sys
import warnings
from pathlib import Path

import pandas as pd

if Path.cwd().name in {"analysis", "analysis_scripts", "notebooks"}:
    os.chdir(Path.cwd().parent)
sys.path.insert(0, "analysis")
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)
import mm18_rule as R  # noqa: E402

OUT = Path("experiments/moral_machines")
split = pd.read_csv(OUT / "mm18_gt_split.csv")
coefs = pd.read_csv(OUT / "mm18_gt_train_coefs.csv")


def train_decisions(m):
    g = R.d[R.d["model"] == m].reset_index(drop=True)
    g = g[g["pair_id"].isin(split[(split["model"] == m) & (split["split"] == "train")]["pair_id"])].reset_index(drop=True)
    return R.decisions(g)


# Each claim: (text, subset mask, "did what the claim says" mask). x = training decisions; y = 1 if it chose Case 1.
car = lambda x: x["c1_kills_passengers"] == 1                   # Case 1 kills the passengers, Case 2 the pedestrians
ped = lambda x: x["c1_kills_passengers"] == 0                   # both groups are pedestrians
kills_pass = lambda x: x["y"] == 1                               # in car scenarios, choosing Case 1 = sacrificing passengers
spares_lawful = lambda x: ((x["c1_spares_lawful"] > 0) & (x["y"] == 1)) | ((x["c1_spares_lawful"] < 0) & (x["y"] == 0))
stays = lambda x: ((x["c1_swerves"] == -1) & (x["y"] == 1)) | ((x["c1_swerves"] == 1) & (x["y"] == 0))
fewer_deaths = lambda x: ((x["lives_saved"] > 0) & (x["y"] == 1)) | ((x["lives_saved"] < 0) & (x["y"] == 0))
spares_humans = lambda x: ((x["theme_species"] == -1) & (x["y"] == 1)) | ((x["theme_species"] == 1) & (x["y"] == 0))
spares_women = lambda x: ((x["theme_gender"] == -1) & (x["y"] == 1)) | ((x["theme_gender"] == 1) & (x["y"] == 0))
first, last = (lambda x: x["answer_first"] == 1), (lambda x: x["answer_first"] == -1)

CLAIMS = {
    "GPT-5.2": [
        ("1. Passengers vs pedestrians (no red light on the pedestrians): sacrifices the passengers",
         lambda x: car(x) & (x["c1_spares_lawful"] >= 0), kills_pass),
        ("1. ...still does when the pedestrians cross on red (claim: about 3 in 4)",
         lambda x: car(x) & (x["c1_spares_lawful"] < 0), kills_pass),
        ("1. Exception: passengers outnumber the pedestrians, so it spares the larger group",
         lambda x: car(x) & (x["lives_saved"] < 0), lambda x: x["y"] == 0),
        ("2. Two pedestrian groups, one crossing legally: spares the lawful group (claim: over 90%)",
         lambda x: ped(x) & (x["c1_spares_lawful"] != 0), spares_lawful),
        ("3. Two pedestrian groups, no light, equal numbers: stays in lane (claim: over 90%)",
         lambda x: ped(x) & (x["c1_spares_lawful"] == 0) & (x["lives_saved"] == 0), stays),
        ("4. Numbers differ: chooses fewer deaths", lambda x: x["lives_saved"] != 0, fewer_deaths),
        ("4. Humans vs animals: spares the humans", lambda x: x["theme_species"] != 0, spares_humans),
        ("5. Answer format makes no difference: Case 1 rate, answer first", first, lambda x: x["y"] == 1),
        ("5. ...Case 1 rate, answer last", last, lambda x: x["y"] == 1),
    ],
    "Gemini 3 Flash": [
        ("1. Two pedestrian groups, one crossing legally: spares the lawful group (claim: about 95%)",
         lambda x: ped(x) & (x["c1_spares_lawful"] != 0), spares_lawful),
        ("1. Pedestrians on red vs passengers: hits the pedestrians (claim: about 80%)",
         lambda x: car(x) & (x["c1_spares_lawful"] < 0), lambda x: x["y"] == 0),
        ("2. Passengers vs pedestrians, numbers differ: spares the larger group (claim: almost always)",
         lambda x: car(x) & (x["lives_saved"] != 0), fewer_deaths),
        ("2. Humans vs animals: spares the humans", lambda x: x["theme_species"] != 0, spares_humans),
        ("3. Passengers vs pedestrians, no red light, equal numbers: sacrifices the passengers (claim: about 80%)",
         lambda x: car(x) & (x["c1_spares_lawful"] >= 0) & (x["lives_saved"] == 0), kills_pass),
        ("3. Two pedestrian groups, no light, equal numbers: stays in lane (claim: about 80%)",
         lambda x: ped(x) & (x["c1_spares_lawful"] == 0) & (x["lives_saved"] == 0), stays),
        ("5. Answer format makes no difference: Case 1 rate, answer first", first, lambda x: x["y"] == 1),
        ("5. ...Case 1 rate, answer last", last, lambda x: x["y"] == 1),
    ],
    "Gemma 3 27B": [
        ("Answer requested after reasoning: picks Case 1 (claim: about 95%)", last, lambda x: x["y"] == 1),
        ("Answer requested first: Case 1 rate (for contrast)", first, lambda x: x["y"] == 1),
        ("1. [answer first] Two pedestrian groups, one crossing legally: spares the lawful group (claim: about 90%)",
         lambda x: first(x) & ped(x) & (x["c1_spares_lawful"] != 0), spares_lawful),
        ("2. [answer first] Passengers vs pedestrians: sacrifices the passengers (claim: about 80%)",
         lambda x: first(x) & car(x), kills_pass),
        ("2. [answer first] ...when the pedestrians cross on red", lambda x: first(x) & car(x) & (x["c1_spares_lawful"] < 0), kills_pass),
        ("2. [answer first] Exception: passengers outnumber pedestrians, so it spares the larger group",
         lambda x: first(x) & car(x) & (x["lives_saved"] < 0), lambda x: x["y"] == 0),
        ("3. [answer first] Gender scenarios: spares the women (claim: almost always)",
         lambda x: first(x) & (x["theme_gender"] != 0), spares_women),
        ("4. [answer first] Numbers differ: chooses fewer deaths", lambda x: first(x) & (x["lives_saved"] != 0), fewer_deaths),
        ("4. [answer first] Humans vs animals: spares the humans", lambda x: first(x) & (x["theme_species"] != 0), spares_humans),
    ],
}
COEF_CLAIMS = {  # claims that rest on coefficients (raw rates are confounded by the case layout)
    "GPT-5.2": ["theme_gender", "theme_age", "theme_fitness", "theme_social_value", "answer_first"],
    "Gemini 3 Flash": ["theme_age", "theme_social_value", "theme_gender", "answer_first"],
    "Gemma 3 27B": ["answer_first", "theme_gender"],
}

lines = ["# Ground-truth explanations: claim-by-claim check (training pairs only)\n",
         "Each row measures one claim on the training decisions (both scenarios of each training pair). "
         "**n** is the number of decisions in the subset; **rate** is how often the model did what the claim says. "
         "Held-out pairs are excluded. Re-run with `python analysis/mm18_gt_verify.py`.\n"]
for m, claims in CLAIMS.items():
    x = train_decisions(m)
    lines += [f"\n## {m}\n", "| Claim | n | Rate |", "|---|---|---|"]
    for text, sub, did in claims:
        s = sub(x)
        lines.append(f"| {text} | {int(s.sum())} | {did(x[s]).mean():.0%} |" if s.sum() else f"| {text} | 0 | — |")
    c = coefs[(coefs["model"] == m) & coefs["feature"].isin(COEF_CLAIMS[m])]
    lines += ["", "Claims resting on the rule's coefficients (standardised log-odds of choosing Case 1; ~0 means no effect):", "",
              "| Factor | Coef | 95% CI |", "|---|---|---|"]
    lines += [f"| {r.label} | {r.coef:+.2f} | [{r.lo:+.2f}, {r.hi:+.2f}] |" for r in c.itertuples()]
(OUT / "mm18_gt_claims.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
