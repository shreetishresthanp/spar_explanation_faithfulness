# %% [markdown]
# # Check the coefficient-informed texts: every percentage is in the model's saved training statistics, ≤150 words
#
# Run from the repo root: `python analysis/mm18_induce_check_texts.py`. No API calls. Exits non-zero on a failure.

# %%
import json
import os
import re
import sys
from pathlib import Path

if Path.cwd().name in {"analysis", "analysis_scripts", "notebooks"}:
    os.chdir(Path.cwd().parent)

OUT = Path("experiments/moral_machines")
MAX_WORDS = 150
texts = {m: v for m, v in json.loads((OUT / "mm18_induce_coef_texts.json").read_text()).items() if not m.startswith("_")}
material = (OUT / "mm18_induce_coef_material.md").read_text()
sections = dict(re.findall(r"^## (.+?)\n(.*?)(?=^## |\Z)", material, re.S | re.M))

ok = True
for m, t in texts.items():
    stats = sections[m].split("### Training statistics")[1].split("### 40 training decisions")[0]
    allowed = set(re.findall(r"(\d+)% of", stats))
    cited = re.findall(r"(\d+)%", t["text"])
    missing = [c for c in cited if c not in allowed]
    words = len(t["text"].split())
    good = not missing and words <= MAX_WORDS and 3 <= len(t["factors"]) <= 5
    ok &= good
    print(f"{'ok ' if good else 'BAD'} {m}: {words} words, {len(t['factors'])} factors, {len(cited)} percentages"
          + (f", not in statistics: {missing}" if missing else ""))
sys.exit(0 if ok else 1)
