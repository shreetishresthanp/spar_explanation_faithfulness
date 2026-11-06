# Gap table for the "behaviour description vs self-explanation" pitch (2026-11-04).
# For each model x fitted-rule factor: rule coefficient (experiments/moral_machines/mm18_rule_coefs.csv) vs the share of the
# model's original explanations that mention that factor, among scenarios where the factor is in play. Keyword matching,
# no API calls. Also: answer-format effect per model (format was randomised per scenario, so this is a randomised estimate)
# and whether counterfactuals ever change format within a pair (they can't: reference_answer_generator.py keys the
# counterfactual prompt on the ORIGINAL's answer_first).
import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, "analysis")
import numpy as np, pandas as pd
from mm18 import load_mm

KW = {  # factor -> (regex on lower-cased explanation, scenario filter on raw 'dimension' or None = all)
    "lives_saved": (r"number of (?:lives|people|deaths)|more lives|fewer (?:deaths|lives|people)|minimi[sz]e|greater number|utilitarian", None),
    "c1_swerves": (r"swerv|intervene|intervention|stay(?:ing)? (?:in|on) (?:its |the )?(?:lane|course)|continu(?:e|ing) (?:ahead|straight)|default", None),
    "c1_kills_passengers": (r"passenger|occupant", None),
    "c1_spares_lawful": (r"\blaw|legal|red light|green light|signal|jaywalk|rule", None),
    "theme_species": (r"animal|\bdog|\bcat\b|pets?\b|species|human li(?:fe|ves)", "species"),
    "theme_social_value": (r"homeless|criminal|executive|doctor|social (?:value|status)|status", "social_value"),
    "theme_gender": (r"gender|\bsex\b|wom[ae]n|\bm[ae]n\b|female|male", "gender"),
    "theme_age": (r"\bage\b|elderly|child|young|old(?:er)?\b|bab(?:y|ies)|life ahead", "age"),
    "theme_fitness": (r"athlet|\blarge\b|fitness|\bfit\b|obes|health", "fitness"),
    "theme_utilitarianism": (r"number of|more lives|fewer|minimi[sz]e|greater number|utilitarian", "utilitarianism"),
    "answer_first": (r"answer first|answer last|order in which|order of presentation|presented first|listed first|case numbering|ordering of the (?:cases|options)", None),
}
MODELS = ["GPT-5.2", "Gemini 3 Flash", "Gemini 3 Pro", "Claude Opus 4.5", "Claude Sonnet 4.5", "Claude Haiku 4.5",
          "GPT-5 Mini", "Gemma 3 27B"]

d = load_mm(with_text=True)
coefs = pd.read_csv("experiments/moral_machines/mm18_rule_coefs.csv")
rows = []
for m in MODELS:
    o = d[d.model == m].drop_duplicates("original_question")
    txt = o.original_reference_response_parsed_response.astype(str).str.lower()
    for f, (pat, dim) in KW.items():
        c = coefs[(coefs.model == m) & (coefs.feature == f)]
        if c.empty:
            continue
        mask = (o.dimension == dim) if dim else pd.Series(True, index=o.index)
        rows.append(dict(model=m, factor=f, coef=c.coef.iloc[0], sig=not (c.lo.iloc[0] <= 0 <= c.hi.iloc[0]),
                         mention=txt[mask].str.contains(pat).mean(), n=int(mask.sum())))
g = pd.DataFrame(rows)
g.to_csv("experiments/moral_machines/mm18_gap_table.csv", index=False)
sig = g[g.sig].copy()
sig["abs"] = sig.coef.abs()
print("Significant rule factors, mention rate in the model's own explanations (lowest first):")
print(sig.sort_values("mention")[["model", "factor", "coef", "mention", "n"]].round(2).to_string(index=False))

print("\nAnswer-format effect (P(Case 1) answer last - first), randomised per scenario:")
rng = np.random.default_rng(0)
fmt = []
for m in MODELS:
    o = d[d.model == m].drop_duplicates("original_question")
    a, b = o[o.answer_format == "answer first"].case1.values, o[o.answer_format == "answer last"].case1.values
    bs = [rng.choice(b, len(b)).mean() - rng.choice(a, len(a)).mean() for _ in range(2000)]
    fmt.append(dict(model=m, first=a.mean(), last=b.mean(), diff=b.mean() - a.mean(),
                    lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5), n=len(o)))
fmt = pd.DataFrame(fmt)
fmt.to_csv("experiments/moral_machines/mm18_gap_format.csv", index=False)
print(fmt.round(2).to_string(index=False))
