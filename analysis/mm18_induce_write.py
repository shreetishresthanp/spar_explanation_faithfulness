# %% [markdown]
# # Induced ground truth: free-form drafts written by an outside observer (exploratory)
#
# Run from the repo root after `python analysis/mm18_induce_prep.py`:
#   python analysis/mm18_induce_write.py --dry-run     # print one rendered prompt, its size and the projected cost
#   python analysis/mm18_induce_write.py               # 9 models x 5 drafts (resumable)
# The writer (Claude Opus 5.5 via OpenRouter) gets the identical prompt for every model
# (analysis/prompts/mm18_induce_writer_prompt.txt) plus that model's training pairs
# (experiments/moral_machines/mm18_induce_writer_input/<model>.txt). No coefficients, statistics or explanations.
# Writes to experiments/moral_machines/:
# - mm18_induce_drafts.json: {model: [{draft, text, factors, words}, ...]} (accepted drafts only)
# - mm18_induce_writer_calls.csv: every call, including rejected attempts, with raw output, tokens and cost

# %%
import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

import httpx
import pandas as pd

if Path.cwd().name in {"analysis", "analysis_scripts", "notebooks"}:
    os.chdir(Path.cwd().parent)
sys.path.insert(0, ".")
from src.utils import api_key  # noqa: E402

OUT = Path("experiments/moral_machines")
PROMPT = Path("analysis/prompts/mm18_induce_writer_prompt.txt").read_text()
WRITER, TEMPERATURE, MAX_TOKENS = "anthropic/claude-opus-5.5", 1.0, 4000
N_DRAFTS, MAX_ATTEMPTS, MAX_WORDS = 5, 3, 150
MODELS = ["GPT-5.2", "GPT-5 Mini", "GPT-5 Nano", "Claude Opus 4.5", "Claude Sonnet 4.5", "Claude Haiku 4.5",
          "Gemini 3 Pro", "Gemini 3 Flash", "Gemma 3 27B"]
DRAFTS, CALLS = OUT / "mm18_induce_drafts.json", OUT / "mm18_induce_writer_calls.csv"


def render(model):
    return PROMPT.replace("{decisions}", (OUT / "mm18_induce_writer_input" / f"{model.replace(' ', '_')}.txt").read_text())


def parse(content):
    """JSON object from the reply (tolerates a code fence); None if it isn't valid."""
    m = re.search(r"\{.*\}", content or "", re.S)
    try:
        d = json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None
    if not isinstance(d, dict) or not isinstance(d.get("text"), str) or not isinstance(d.get("factors"), list):
        return None
    return d


async def write_one(client, prompt):
    r = await client.post("https://openrouter.ai/api/v1/chat/completions", timeout=600, json=dict(
        model=WRITER, temperature=TEMPERATURE, max_tokens=MAX_TOKENS, messages=[{"role": "user", "content": prompt}],
        usage={"include": True}))
    r.raise_for_status()
    o = r.json()
    return (o["choices"][0]["message"].get("content") or ""), o.get("usage") or {}


async def draft(client, sem, model, k):
    """One accepted draft: retries (up to MAX_ATTEMPTS) only for invalid JSON or > MAX_WORDS words; logs every attempt."""
    prompt, log = render(model), []
    for attempt in range(1, MAX_ATTEMPTS + 1):
        async with sem:
            content, usage = await write_one(client, prompt)
        d = parse(content)
        words = len(d["text"].split()) if d else None
        ok = d is not None and words <= MAX_WORDS and 3 <= len(d["factors"]) <= 5
        log.append(dict(model=model, draft=k, attempt=attempt, accepted=ok, words=words, content=content,
                        prompt_tokens=usage.get("prompt_tokens"), completion_tokens=usage.get("completion_tokens"),
                        cost=usage.get("cost")))
        if ok:
            return dict(draft=k, text=d["text"], factors=d["factors"], words=words, attempts=attempt), log
    return None, log


async def main(todo):
    sem = asyncio.Semaphore(9)
    async with httpx.AsyncClient(headers={"Authorization": f"Bearer {api_key}"}) as client:
        return await asyncio.gather(*[draft(client, sem, m, k) for m, k in todo])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--budget", type=float, default=20.0, help="refuse to start if the projected cost is above this ($)")
    args = ap.parse_args()
    assert api_key, "OPENROUTER_API_KEY missing from .env"
    price = {m["id"]: m["pricing"] for m in httpx.get("https://openrouter.ai/api/v1/models", timeout=30).json()["data"]}[WRITER]
    drafts = {m: v for m, v in json.loads(DRAFTS.read_text()).items() if not m.startswith("_")} if DRAFTS.exists() else {}
    todo = [(m, k) for m in MODELS for k in range(1, N_DRAFTS + 1) if k not in {x["draft"] for x in drafts.get(m, [])}]
    chars = sum(len(render(m)) for m, _ in todo)
    projected = chars / 4 * float(price["prompt"]) + len(todo) * 1500 * float(price["completion"])
    print(f"{len(todo)} drafts to write with {WRITER}; ~{chars / 4 / 1e6:.2f}M input tokens; projected ${projected:.2f} "
          "(before any retries)")
    if args.dry_run:
        print(render(MODELS[0])[:3000])
        sys.exit(0)
    assert projected <= args.budget, f"projected ${projected:.2f} is over the ${args.budget:.2f} cap"
    results = asyncio.run(main(todo))
    logs = [row for _, log in results for row in log]
    pd.DataFrame(logs).to_csv(CALLS, mode="a", header=not CALLS.exists(), index=False)
    for (m, _), (d, _) in zip(todo, results):
        if d:
            drafts.setdefault(m, []).append(d)
    for m in drafts:
        drafts[m].sort(key=lambda x: x["draft"])
    DRAFTS.write_text(json.dumps({"_note": f"Free-form induced descriptions. Writer {WRITER}, temperature {TEMPERATURE}; "
                                           "prompt analysis/prompts/mm18_induce_writer_prompt.txt; input "
                                           "mm18_induce_writer_input/<model>.txt (training pairs only).",
                                  **{m: drafts[m] for m in MODELS if m in drafts}}, indent=2, ensure_ascii=False))
    calls = pd.read_csv(CALLS)
    print(f"accepted {sum(d is not None for d, _ in results)} of {len(todo)}; attempts this run {len(logs)}; "
          f"cost this run ${pd.DataFrame(logs)['cost'].sum():.2f}; all writer calls ${calls['cost'].sum():.2f}")
    print(calls.groupby("model")["words"].describe()[["min", "max"]].to_string())
