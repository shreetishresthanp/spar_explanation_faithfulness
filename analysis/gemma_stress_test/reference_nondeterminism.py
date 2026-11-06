# %% [markdown]
# # How non-deterministic is a real reference model? Gemma 3 27B re-answering identical prompts
#
# Run from the repo root in the project env: `python analysis/gemma_stress_test/reference_nondeterminism.py [strict]`.
# No API calls: reads the paired test's saved calls (paired_format_test.py --run), where every Moral Machines scenario
# (1,000 original + 1,000 counterfactual) was asked SAMPLES times with the identical prompt and the paper's settings
# (temperature 1.0, top_p 0.95, top_k 64, seeds 0-2). Answers parsed as in that script (lenient by default).
# Per set (original / counterfactual question) and answer position:
#   disagreement = share of sample pairs on the same prompt that give different answers (unbiased for 2q(1-q), where q
#                  is the chance of the scenario's less likely answer);
#   accuracy ceiling from non-determinism alone, for a simulator scored against one sampled answer:
#     lower end  1 - disagreement           (simulator predicts like another sample from the reference);
#     upper end  1 - q, q = (1 - sqrt(1 - 2 * disagreement)) / 2   (simulator always predicts the more likely
#                answer; q from the mean disagreement assumes q is the same across scenarios, so it is approximate).
# Intervals: 95% bootstrap over scenarios.
# Writes experiments/moral_machines/gemma_stress_test/reference_nondeterminism_<parsing>.csv and
# figures/moral_machines/gemma_stress_test/reference_nondeterminism_<parsing>.png.

# %%
import itertools
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paired_format_test as pft  # noqa: E402  (sets the working directory to the repo root)

PARSING = sys.argv[1] if len(sys.argv) > 1 else "lenient"
GROUPS = [("original", "first"), ("original", "last"), ("counterfactual", "first"), ("counterfactual", "last")]


def pair_disagreement(answers):
    pairs = list(itertools.combinations(answers, 2))
    return np.mean([a != b for a, b in pairs]) if pairs else np.nan


def ceiling_upper(d):
    return 1 - (1 - np.sqrt(np.clip(1 - 2 * d, 0, None))) / 2


# %%
c = pft.load_calls()
if PARSING == "lenient":
    c["answer"] = [pft.lenient(r, a) for r, a in zip(c["raw"], c["answer"])]
per = (c.dropna(subset=["answer"]).groupby(["set", "format", "idx"])["answer"]
       .apply(lambda a: pair_disagreement(list(a))).dropna().rename("d").reset_index())

rng = np.random.default_rng(pft.SEED)
rows = []
for (s, f) in GROUPS + [("both", "both"), ("original", "both"), ("counterfactual", "both")]:
    sub = per[((per["set"] == s) | (s == "both")) & ((per["format"] == f) | (f == "both"))]["d"].values
    reps = np.array([sub[rng.integers(0, len(sub), len(sub))].mean() for _ in range(pft.N_BOOT)])
    d, (lo, hi) = sub.mean(), np.percentile(reps, [2.5, 97.5])
    rows.append({"set": s, "format": f, "n_scenarios": len(sub), "disagreement": d, "disagreement_lo": lo,
                 "disagreement_hi": hi, "ceiling_low": 1 - d, "ceiling_high": ceiling_upper(d),
                 "ceiling_low_ci": f"{1 - hi:.3f}-{1 - lo:.3f}",
                 "ceiling_high_ci": f"{ceiling_upper(hi):.3f}-{ceiling_upper(lo):.3f}"})
out = pd.DataFrame(rows)
out.to_csv(pft.OUT / f"reference_nondeterminism_{PARSING}.csv", index=False)
pd.set_option("display.width", 200)
print(f"{PARSING} parsing; {len(c)} calls, {pft.SAMPLES} samples per prompt\n")
print(out.round(3).to_string(index=False))

# %% figure
fig, ax = plt.subplots(figsize=(8, 4.4))
labels, ys = [], []
for i, (s, f) in enumerate(GROUPS + [("both", "both")]):
    r = out[(out["set"] == s) & (out["format"] == f)].iloc[0]
    ax.plot([r["ceiling_low"] * 100, r["ceiling_high"] * 100], [i, i], color="#2a78d6", lw=6, solid_capstyle="butt")
    ax.text(r["ceiling_high"] * 100 + 0.3, i, f"{r['ceiling_low']:.0%}-{r['ceiling_high']:.0%}  "
            f"(pairs disagree {r['disagreement']:.1%})", va="center", fontsize=9, color=pft.MUTED)
    labels.append("all prompts" if s == "both" else f"{s} question, answer {f}")
    ys.append(i)
ax.set_yticks(ys, labels)
ax.invert_yaxis()
ax.set_xlim(80, 106)
ax.axvline(100, color=pft.INK, lw=0.8)
ax.set_xlabel("Accuracy ceiling from reference non-determinism alone (%)")
ax.set_title(f"Gemma 3 27B re-answering identical Moral Machines prompts ({PARSING} parsing)")
fig.tight_layout()
fig.savefig(pft.FIG / f"reference_nondeterminism_{PARSING}.png", bbox_inches="tight")
plt.close(fig)
print(f"\nWrote {pft.OUT / f'reference_nondeterminism_{PARSING}.csv'} and "
      f"{pft.FIG / f'reference_nondeterminism_{PARSING}.png'}")
