# MultiLexNorm 2026 LLM Pipeline

## Pipeline overview

### Prepare Detector
Stage 1-1:
- prepare detector training data
- train an XLM-R detector

### Run Pipeline
Stage 1-2: 
- apply the detector and select normalization candidates from its confidence scores
- support both binary (`O`/`NORM`) and length-aware detector labels

Stage 2: 
- build the Most Frequent Replacement (MFR) dictionary
- read detector predictions and keep tokens with `P(NORM) >= DETECTOR_THRESHOLD`
- apply dictionary with `dictionary_entropy <= ENTROPY_THRESHOLD` filtering -> the rest is LLM candidates

Stage 3:
- build LLM prompts with few-shot examples
- apply LLM

## Data split

weerayut/multilexnorm2026-dev-pub -> Train Split -> TRAIN Data (90%) + DEV Data (10%)

TRAIN:
  - train detector
  - build lookup dictionary
  - sample few-shot examples for LLM prompts

DEV:
  - apply detector
  - apply dictionary
  - apply LLM
  - evaluate outputs

## Files
### Config
- `models/machamp/configs/machamp_detector_en.json`: config file for MaChAmp dataset
- `models/machamp/configs/machamp_params_detector.json`: config file for MaChAmp training
- `src/config.py`: config file for hyperparameters and file paths

### Scripts
- `src/inspect_hf_dataset.py`: helper function: inspect and export the HF dataset
- `src/prepare_detector_data.py`: convert HF or synthetic data to detector training data and generate a language-specific MaChAmp config
- `src/import_clean_wiki.py`: import, clean, and tokenize Wikipedia text
- `src/add_noise.py`: create synthetic normalization data with learned and rule-based noise
- `src/execute_pretrain_finetune_experiment.py`: run synthetic pretraining, original-data finetuning, and dev prediction
- `src/execute_prepare_detector.py`: execution script: prepare detector data and train the detector.
- `src/build_dictionary.py`: build MFR lookup dictionary candidates
- `src/apply_dictionary.py`: apply dictionary replacements to detector labeled tokens
- `src/build_llm_prompts.py`: build LLM prompts for LLM candidates.
- `src/run_llm.py`: run LLM inference and merge results.
- `src/evaluate_pipeline.py`: evaluate final pipeline results.
- `src/execute_run_pipeline.py`: execution script: run the 2026 LLM pipeline end to end.
- `src/search_thresholds.py`: run a real detector and entropy threshold grid search with reusable caches.
- `src/plot_threshold_results.py`: plot final ERR and F1 from threshold search results.
- `src/execute_full_search_thresholds.py`: run threshold search for multiple languages.
- `src/execute_val_threshold.py`: evaluate original or new detector outputs on validation data with selected thresholds.

### Example data

Small sample files under `examples/`.

- `examples/detector_train/detector_train_sample.tsv`: detector TRAIN data.
- `examples/detector_train/detector_dev_sample.tsv`: detector DEV data.
- `examples/detector_output/detector_confidence_sample.out`: detector confidence predictions.
- `examples/detector_output/detector_en.confidence.out.eval`: detector model evaluation result
- `examples/dictionary_sample.jsonl`: MFR dictionary
- `examples/table_applied_dictionary_sample.jsonl`: master table after applying dictionary
- `examples/llm_candidates_sample.jsonl`: remaining LLM candidates list
- `examples/llm_prompts_sample.jsonl`: LLM prompts for each LLM candidate
- `examples/llm_candidates_applied_llm_sample.jsonl`: LLM-updated candidate records.
- `examples/table_applied_llm_sample.jsonl`: final master table after applying LLM

# Usage

## 1. Download

### Dependencies

```bash
cd llm_pipeline
conda create -n llm-pipeline python=3.10 -y
conda activate llm-pipeline
pip install -r requirements.txt
```

### Machamp Toolkit

```bash
mkdir -p external
git clone https://github.com/machamp-nlp/machamp.git external/machamp
pip install -r external/machamp/requirements.txt
```

### Ollama Model

```bash
ollama pull qwen3.5:9b
ollama list
```

### DeepSeek API

Set the DeepSeek API key:

```bash
export DEEPSEEK_API_KEY="your-api-key"
```

The model is selected with the `MODEL` setting in `src/config.py` or the
`--model` command-line option. Models whose names start with `deepseek-` use
the DeepSeek API; all other model names use the local Ollama API.

## 2. Run

### Check Default Language and Thresholds

Open `src/config.py`:

```bash
LANGUAGE = os.getenv("PIPELINE_LANGUAGE", "en")
DETECTOR_THRESHOLD = float(os.getenv("PIPELINE_DETECTOR_THRESHOLD", "0.5"))
ENTROPY_THRESHOLD = float(os.getenv("PIPELINE_ENTROPY_THRESHOLD", "0.5"))
MODEL = os.getenv("MODEL", "qwen3.5:9b")

```

### Model Preparation

Prepare detector data and train the detector:

```bash
python -m src.execute_prepare_detector --language en
```

### Run Pipeline

Run detector prediction, build dictionary, apply dictionary, build LLM
prompts, run LLM inference, and evaluate:

```bash
python -m src.execute_run_pipeline \
  --language en \
  --detector-threshold 0.5 \
  --entropy-threshold 0.5 \
  --model qwen3.5:9b
```

Pipeline outputs dir:

```text
data/qwen3.5:9b/en_0.5_0.5/
```

## 3. Analyze Thresholds

### Grid Threshold Search

The default grid contains:

- Detector thresholds: `0.1, 0.3, 0.5, 0.7, 0.9`
- Dictionary entropy thresholds: `0.2, 0.5, 0.8, 1.1, 1.4, 1.7`

The first run builds reusable caches. External detector output and dictionary
files can also be supplied.

Run a threshold search for one language:

```bash
python -m src.search_thresholds \
  --language en \
  --model qwen3.5:9b \
  --detector-output \
  ../our_pipeline1/data/detector_output/detector_en.confidence.out \
  --dictionary ../our_pipeline1/data/dictionary_en.jsonl \
  --output-root data/newdetector_qwen3.5:9b
```

The selected thresholds, evaluation results, and ERR/F1 plots are written to:

```text
data/newdetector_qwen3.5:9b/en_thresholds/
```

Run batch threshold grid search for multiple languages:

```bash
python -m src.execute_full_search_thresholds \
  --model qwen3.5:9b \
  --languages de en hr id iden ja ko nl sl sr th vi
```

### Evaluate a Selected Validation Threshold

Run the official validation data with either the original or new detector. The
`original` detector reads from `../new_pipeline/data` by default; the `new`
detector reads from `../our_pipeline1/data` by default. Use `--original-data`
or `--new-data` to override those roots.

```bash
python -m src.execute_val_threshold \
  --detector new \
  --model qwen3.5:9b \
  --languages en \
  --threshold 0.5 0.5
```

Add `--prepare-only` to prepare tables and prompts without running the LLM.

# Appendix

## Pipeline Steps

1. Prepare training data:

```bash
python -m src.prepare_detector_data
```

This creates:

```text
MaChAmp dataset config: models/machamp/configs/machamp_detector_en.json
Data for training: models/machamp/train_dev/en/detector_train_en.tsv
Data for evaluation: models/machamp/train_dev/en/detector_dev_en.tsv
```


2. Train detector:

```bash
python external/machamp/train.py \
  --dataset_configs models/machamp/configs/machamp_detector_en.json \
  --parameters_config models/machamp/configs/machamp_params_detector.json \
  --model_dir models/machamp/detector_en_xlmr \
  --device 0
```

3. Predict on the dev split with detector confidence:

```bash
mkdir -p data/qwen3.5:9b/en_0.5_0.5/detector_output
python external/machamp/predict.py \
  models/machamp/detector_en_xlmr/model.pt \
  models/machamp/train_dev/en/detector_dev_en.tsv \
  data/qwen3.5:9b/en_0.5_0.5/detector_output/detector_en.confidence.out \
  --dataset detector_en \
  --device 0 \
  --topn 2
```

This creates:

```text
data/qwen3.5:9b/en_0.5_0.5/detector_output/detector_en.confidence.out
data/qwen3.5:9b/en_0.5_0.5/detector_output/detector_en.confidence.out.eval
```

4. Build MFR dictionary from TRAIN data:

```bash
python -m src.build_dictionary
```

This creates:

```text
data/qwen3.5:9b/en_0.5_0.5/dictionary_en.jsonl
```

5. Apply dictionary lookup to detector predictions:

```bash
python -m src.apply_dictionary
```

This creates:

```text
data/qwen3.5:9b/en_0.5_0.5/table_applied_dictionary_en.jsonl
data/qwen3.5:9b/en_0.5_0.5/llm_candidates_en.jsonl
```

6. Build LLM prompts for remaining candidates:

```bash
python -m src.build_llm_prompts
```

This creates:

```text
data/qwen3.5:9b/en_0.5_0.5/llm_prompts_en.jsonl
```

7. Run LLM inference

With Ollama:

```bash
python -m src.run_llm --model qwen3.5:9b
```

With DeepSeek:

```bash
export DEEPSEEK_API_KEY="your-api-key"
python -m src.run_llm --model deepseek-v4-pro
```

`src.run_llm` reads and writes all LLM files in the current experiment
directory. Threshold-search caches are managed internally by
`src.search_thresholds`.

This creates:

```text
data/qwen3.5:9b/en_0.5_0.5/llm_candidates_applied_llm_en.jsonl
data/qwen3.5:9b/en_0.5_0.5/table_applied_llm_en.jsonl
```

8. Evaluation

Evaluate the final result:

```bash
python -m src.evaluate_pipeline \
  --language en \
  --model qwen3.5:9b \
  --detector-threshold 0.5 \
  --entropy-threshold 0.5
```

This creates:

```text
data/qwen3.5:9b/en_0.5_0.5/evaluation_summary_en.json
```

## Synthetic Data Pretraining

This experiment imports and tokenizes clean Wikipedia text, adds noise learned
from the original training data together with simple character-level rules,
then pretrains on the synthetic pairs and finetunes on the original train
split.

```bash
python -m src.execute_pretrain_finetune_experiment \
  --language en \
  --run-name wiki5k \
  --max-sentences 5000 \
  --device 0
```

Data and models are written under `data/wiki_pretrain_finetune/` and
`models/machamp/`. Default Wikipedia URLs are available for `en`, `de`, `it`,
`ja`, `ko`, and `nl`.

## Master Table

`data/qwen3.5:9b/en_0.5_0.5/table_applied_dictionary_en.jsonl` and
`data/qwen3.5:9b/en_0.5_0.5/table_applied_llm_en.jsonl` use
one JSON object per token:

```json
{
  "Token_id": 22,
  "Sentence_id": 1,
  "Token_index": 4,
  "RAW": "d",
  "Gold_NORM": "the",
  "Detector_label": "NORM",
  "Detector_confidence": 0.999913215637207,
  "Detector_norm_confidence": 0.999913215637207,
  "Detector_format": "binary",
  "Replacement": "the",
  "Dictionary_entropy": null,
  "Source": "qwen3.5:9b"
}
```

- `Token_id`: global token id across DEV data.
- `Sentence_id`: sentence id in DEV data.
- `Token_index`: token position inside the sentence.
- `RAW`: original token.
- `Gold_NORM`: gold normalized token for evaluation.
- `Detector_label`: detector decision, `O`, `NORM`, or a length-aware NORM label.
- `Detector_confidence`: confidence score of `Detector_label`.
- `Detector_norm_confidence`: total normalization probability used for thresholding.
- `Detector_format`: detector output format, `binary` or `length_aware`.
- `Replacement`: current pipeline prediction.
- `Dictionary_entropy`: dictionary entropy if `Source` is `dictionary`, otherwise `null`.
- `Source`: replacement source, such as `keep`, `dictionary`, `llm_pending`, or the LLM model name.
