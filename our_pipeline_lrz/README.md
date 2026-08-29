# Our Pipeline

> [!IMPORTANT]
> This folder contains the **LRZ-only experimental version** of the pipeline.
> Its SLURM resources, container image, Conda environment, project paths, and
> local model paths are specific to LRZ. All supported experiment workflows are
> packaged as top-level `.sh` scripts and should normally be submitted with
> `sbatch`.

## Pipeline overview

The LRZ workflow consists of four stages:

1. Train a Length-Aware Detector with the labels `O`, `NORM_1WORD`,
   `NORM_2WORD`, and `NORM_3PLUS`.
2. Build the most-frequent-replacement (MFR) dictionary and prepare the
   remaining normalization candidates for LLM inference.
3. Search for language-specific detector and dictionary-entropy thresholds on
   the development split.
4. Run and evaluate the fixed-threshold or threshold-optimized pipeline on the
   validation split.

The public training split is divided into 90% training data and 10%
development data. The training portion is used to train the detector, build the
MFR dictionary, and sample few-shot examples. The development portion is used
for threshold search. Final experiments use the separate validation split.

## LRZ prerequisites

Run all commands from `our_pipeline_lrz`. Before submitting jobs, edit the LRZ
paths near the top of each script if the project, container, Conda environment,
dataset, or local model is stored elsewhere.

```bash
cd /dss/dsshome1/01/ge65nus2/projects/MultiLexNorm_LLM/our_pipeline_lrz
```

The scripts expect:

- the Conda environment `new_pipeline`;
- MaChAmp under `external/machamp`;
- the LRZ PyTorch container referenced by the `#SBATCH` directives;
- locally available Hugging Face model weights for threshold search and
  validation inference;
- the dataset saved at the LRZ path used by
  `src/prepare_validation_data.py`.

### Detector and LLM package versions

The versions in `requirements.txt` describe the environment used to prepare
and train the detector. In particular, that environment contains
`transformers==4.57.6` and `huggingface_hub==0.36.2`.

Local LLM inference was run after replacing **only** these two packages with
the following versions:

```bash
python -m pip install --no-deps --force-reinstall \
  transformers==5.8.1 \
  huggingface-hub==1.14.0
```

The `--no-deps` option is required: it prevents pip from upgrading,
downgrading, or reinstalling any other package in the detector environment.
Run this command after detector preparation and before
`run_threshold_search.sh`, `run_evaluate_val.sh`, or
`run_evaluate_val_op.sh`, all of which load the local LLM directly or through
`src/run_llm.py`.

Verify the active versions before submitting an LLM job:

```bash
python -c "import transformers, huggingface_hub; \
print(transformers.__version__, huggingface_hub.__version__)"
```

## Shell-script workflow

The scripts below are ordered by their role in the experimental workflow.

### 1. `run_train_detector.sh`

Prepares detector train/development data for all languages, trains one MaChAmp
Length-Aware Detector per language, predicts development-set confidences, builds
the MFR dictionaries, and prepares development-set LLM candidates and prompts.

Main outputs for each `<lang>` are:

```text
models/machamp/configs/machamp_detector_<lang>.json
models/machamp/detector_<lang>_xlmr/model.pt
data/detector_train/detector_train_<lang>.tsv
data/detector_train/detector_dev_<lang>.tsv
data/detector_output/detector_<lang>.confidence.out
data/<lang>/dictionary_<lang>.jsonl
data/<lang>/table_applied_dictionary_<lang>.jsonl
data/<lang>/llm_candidates_<lang>.jsonl
data/<lang>/llm_prompts_<lang>.jsonl
```

```bash
sbatch run_train_detector.sh
```

### 2. `run_threshold_search.sh`

Runs the detector-threshold and dictionary-entropy-threshold grid search on the
development split for the selected Hugging Face model. The search is resumable
and stores per-configuration intermediate results.

Main outputs are:

```text
data/threshold_search_results_<MODEL_NAME>.jsonl
data/thresold_search_tmp/<MODEL_NAME>/<lang>/grid_results.jsonl
```

```bash
sbatch run_threshold_search.sh
```

### 3. `run_prepare_val.sh`

Converts the validation split into detector input and gold files, applies each
trained detector, and prepares the fixed-threshold dictionary tables, LLM
candidates, and prompts. This script does not run the LLM.

Main outputs for each `<lang>` are:

```text
data/detector_train/detector_val_<lang>.tsv
data/detector_output/detector_<lang>_val.confidence.out
data/<lang>/gold_<lang>_val.jsonl
data/<lang>/table_applied_dictionary_<lang>_val.jsonl
data/<lang>/llm_candidates_<lang>_val.jsonl
data/<lang>/llm_prompts_<lang>_val.jsonl
```

```bash
sbatch run_prepare_val.sh
```

### 4. `run_evaluate_val.sh`

Runs validation inference and evaluation with the Length-Aware Detector and the
default detector and entropy thresholds. It rebuilds the fixed-threshold
candidate files before running the selected local Hugging Face model.

Main outputs for each `<lang>` are:

```text
data/<lang>/llm_candidates_applied_llm_<lang>_val.jsonl
data/<lang>/table_applied_<MODEL_NAME>_<lang>_val.jsonl
data/<lang>/evaluation_summary_<MODEL_NAME>_<lang>_val.json
```

```bash
sbatch run_evaluate_val.sh
```

### 5. `run_evaluate_val_op.sh`

Runs the complete validation pipeline with the Length-Aware Detector and the
language-specific thresholds selected by `run_threshold_search.sh`. The script
requires `data/threshold_search_results_<MODEL_NAME>.jsonl`.

Main outputs for each `<lang>` are:

```text
data/<lang>/llm_candidates_applied_llm_<lang>_val_OP.jsonl
data/<lang>/table_applied_<MODEL_NAME>_<lang>_val_OP.jsonl
data/<lang>/evaluation_summary_<MODEL_NAME>_<lang>_val_OP.json
```

```bash
sbatch run_evaluate_val_op.sh
```

## Configurable variables

The shell scripts set the relevant variables before importing `src/config.py`.
Edit the assignments near the top of a script when changing an experiment.

### LLM configuration

- `MODEL_NAME`: filesystem-safe model identifier used in result filenames and
  threshold-search files, for example `qwen3.5_9b`.
- `HF_MODEL_PATH`: LRZ path to the local Hugging Face model weights.
- `DTYPE` / `LLM_DTYPE`: inference precision, currently `float16` or
  `bfloat16`.

`MODEL_NAME` must match between threshold search and optimized validation so
that `run_evaluate_val_op.sh` loads the correct threshold file.

### Dataset and optimization switches

- `IS_VAL=true` selects validation input and adds `_val` to output filenames.
  If it is unset or false, the development split is used.
- `IS_OPTIMIZED=true` enables threshold optimization, loads the selected
  thresholds from `data/threshold_search_results_<MODEL_NAME>.jsonl`, and adds
  `_OP` to the output suffix. If it is unset or false, the default thresholds
  are used. This is the pipeline's **OP switch**; there is no separate variable
  named `OP`.
- `LANGUAGES` controls which language jobs are executed. `LANGUAGE` is exported
  by each loop for the Python modules.

The resulting output suffixes are:

| `IS_VAL` | `IS_OPTIMIZED` | Output suffix |
|---|---|---|
| false | false | no suffix (development) |
| true | false | `_val` |
| true | true | `_val_OP` |

`DETECTOR_THRESHOLD` and `ENTROPY_THRESHOLD` can also be exported to override
individual values explicitly. When `IS_OPTIMIZED=true`, leave these variables
unset so that the values selected by threshold search are loaded instead.

## Implementation reference

- `src/config.py`: environment variables, thresholds, and generated paths.
- `src/prepare_detector_data.py`: detector labels and MaChAmp training data.
- `src/prepare_validation_data.py`: validation detector input and gold data.
- `src/build_dictionary.py`: MFR dictionary construction.
- `src/apply_dictionary.py`: detector decision and dictionary application.
- `src/build_llm_prompts.py`: few-shot prompt construction.
- `src/run_llm.py`: local Hugging Face LLM inference and output merging.
- `src/search_thresholds.py`: resumable threshold grid search.
- `src/evaluate_pipeline.py`: token-, sentence-, detector-, dictionary-, and
  LLM-level evaluation.

## Master-table format

The dictionary-stage and final master tables use one JSON object per token:

```json
{
  "Token_id": 22,
  "Sentence_id": 1,
  "Token_index": 4,
  "RAW": "could've",
  "Gold_NORM": "could have",
  "Detector_label": "NORM_2WORD",
  "Detector_confidence": 0.91,
  "Detector_norm_confidence": 0.97,
  "Replacement": "could have",
  "Dictionary_entropy": null,
  "Source": "qwen3.5_9b"
}
```

- `Detector_label` is `O`, `NORM_1WORD`, `NORM_2WORD`, or `NORM_3PLUS`.
- `Detector_confidence` is the confidence of the selected detector label.
- `Detector_norm_confidence` is the summed confidence of all non-`O` labels.
- `Replacement` is the current normalization prediction.
- `Dictionary_entropy` is populated for dictionary candidates when available.
- `Source` records whether the output came from `keep`, `dictionary`, the LLM,
  or an intermediate pending stage.
