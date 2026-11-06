# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Research code for the paper *A Positive Case for Faithfulness: LLM Self-Explanations Help Predict Model Behavior* (arXiv 2602.02639). It measures whether a **reference model's** self-explanation helps a **predictor model** guess how the reference model will answer counterfactual versions of a question.

## Environment & commands

```bash
conda env create -f environment.yml && conda activate faithfulness-env
```

Run everything **as a module from the repo root** (`python -m src....`). Imports are absolute from `src.`, and some scripts use repo-relative paths.

- API models go through OpenRouter. `src/utils.py` loads `.env` via `python-dotenv` and reads `OPENROUTER_API_KEY` at import time. If the key is missing it only prints a warning.
- Local models use vLLM and need a GPU. Pick devices with `CUDA_VISIBLE_DEVICES`.

Pipeline (each stage reads a parquet and writes an enriched parquet):

```bash
# 1. Build counterfactual datasets (already committed under data/natural_counterfactuals/)
python -m src.counterfactual_generation.tabular_counterfactual_generation.tabular_to_text --output_dir data/natural_counterfactuals
PYTHONHASHSEED=0 python -m src.counterfactual_generation.tabular_counterfactual_generation.moral_machines_generator
python -m data.natural_counterfactuals.generate_combined

# 2. Reference answers (one model)
python -m src.reference_answer_generation.generate_reference_answers <in.parquet> --output-parquet <out.parquet> --model Qwen/Qwen3-8B
#    or several reference models at once
python experiment_scripts/run_multi_model_experiment.py <in.parquet> --models <m1> <m2> --output-folder <dir>

# 3. Predictor answers (--model takes several; also --predictor-repeats N, --assess-testability)
python -m src.prediction_generation.generate_predictor_answers <ref.parquet> --output-parquet <pred.parquet> --model openrouter/qwen/qwen3-32b

# 4. Analysis (also --normalized, --multi-predictor, --consistency-filter)
python -m analysis_scripts.analyze_simulatability <pred.parquet> --output <results.csv>
```

`data/natural_counterfactuals/combined_dataset_subset_20.parquet` is a small input for smoke runs. `tests/run_tests.sh` holds an end-to-end example invocation of the pipeline.

### Tests

`pytest.ini` sets `testpaths = tests`, `--strict-markers` and verbose output, and registers the markers `slow`, `integration`, `unit`, `data_quality`, `requires_gpu`, `requires_api` and `smoke`.

```bash
pytest                                                    # all
pytest tests/test_sample_data_bank.py::TestClassName -v   # single class
pytest -m "not slow and not requires_gpu"                 # skip heavy tests
```

Some tests read generated files under `tests/test_outputs/`, which is not committed. Create them with `tabular_to_text --output_dir tests/test_outputs` followed by `python -m tests.utils.make_sample`. The README mentions `tests/test_sample_data.py`, but that file doesn't exist; the actual file is `tests/test_sample_data_bank.py`.

## Architecture

**Central data model: `src/schema.py`.** Every stage passes data around as a `CounterfactualDatabase` (a list of `FaithfulnessRecord`) stored as parquet.
- A record is `OriginalQuestion` + `CounterfactualInfo` + optional `MatchInfo`. `Response` and `ModelInfo` are nested inside these.
- `to_flat_dict()` flattens nested dataclasses into prefixed columns, such as `original_reference_response_answer` and `counterfactual_predictor_response_with_explanation_model_info_model`. `load_parquet()` rebuilds the objects by parsing those prefixes.
- **When you add a field to a dataclass, the flatten/unflatten prefix logic must still round-trip it.** Pay particular attention to new nested `Response`/`ModelInfo` fields; their names are hard-coded in both methods.
- `CounterfactualInfo.question_idx` is auto-assigned starting at 100000001 by `add_record`.
- The `src/README.md` schema section is an older flat listing. Treat `schema.py` as the source of truth.

**Dataset templates: `src/templates/`.** Each dataset is a class, mostly subclasses of `TabularDataset` in `base.py`. Each class defines:
- `VALID_ANSWERS`
- `load_dataset`
- `description_generator`
- the three prompt builders: `create_reference_prompt`, `create_counterfactual_prompt` (with the original explanation) and `create_counterfactual_prompt_no_explanation`

Records store a `dataset` string, and stages look up the class through `CounterfactualDatabase.dataset_class_map`. **To add a dataset, register it in that map.** `notebooks/06_adding_new_dataset_example.ipynb` walks through the process.

**Model configs: `src/vllm_configs.py`.** `VLLM_CONFIGS` maps a CLI `--model` string to an `LLMConfig` (defined in `src/utils.py`).
- Every CLI rejects model names that aren't keys in this dict.
- Keys prefixed `openrouter/` have `api_model=True` and are called asynchronously over HTTP with a concurrency semaphore. All other keys are loaded in-process with vLLM.
- `enable_reasoning` is passed through as chat-template kwargs: a bool for Qwen, `"low"`/`"medium"`/`"high"` for gpt-oss.
- `cot_flags` lists the chain-of-thought delimiters, and the **last flag must be the separator** that `split_on_cot_seperator` splits on. gpt-oss output is parsed with Harmony (`parse_message_to_harmony`).

**Generators.**
- `ReferenceAnswerGenerator` (`src/reference_answer_generation/`):
  - deduplicates identical prompts before inference
  - maps the answers back onto records
  - builds `prompt_with_explanation` / `prompt_without_explanation` for each counterfactual from the reference model's answer and explanation on the original question
- `PredictorAnswerGenerator` (`src/prediction_generation/`):
  - answers those two prompts
  - supports several predictor models and repeats, recorded in `Response.predictor_answers` / `predictor_names`
  - can append predictors to a parquet that already has predictions
  - can optionally score counterfactual testability (`testability_rubric.txt`)
  - frees GPU memory between models with `cleanup_after_model`
- Answers are parsed from JSON and checked with `parse_response` / `normalize_answer` against the dataset's `VALID_ANSWERS`. `filter_records_by_reference_answer` drops records whose reference answer is invalid before prediction.

**Counterfactual sources.**
- `tabular_counterfactual_generation/` builds "natural" Hamming-ball counterfactuals by perturbing feature values of tabular rows and rendering them as text. `moral_machines_generator` must be run with `PYTHONHASHSEED=0` so its output is reproducible.
- `llm_counterfactual_generation/` has LLMs generate counterfactuals using the prompt files in `methods/`.

**Analysis.**
- `analysis_scripts/` computes simulatability gain: accuracy with explanation minus accuracy without, plus a normalized gain.
- `analysis_scripts/features/` holds feature-level breakdowns.
- `notebooks/` and `figures/figs.ipynb` produce the paper figures.
- Experiment outputs conventionally go under `experiments/`.
