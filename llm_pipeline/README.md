# MultiLexNorm 2026 LLM Pipeline

## Pipeline overview

### Prepare Detector
Stage 1-1:
- prepare detctor training data
- train a XLM-R detector

### Run Pipeline
Stage 1-2: 
- apply detector to label tokens by 0/NROM confidence score

Stage 2: 
- build MFR dictionary
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
- `src/prepare_detector_data.py`: convert HF data to labeled data for training detector, generate language-specified MaChAmp dataset config
- `src/execute_prepare_detector.py`: execution script: prepare detector data and train the detector.
- `src/build_dictionary.py`: build MFR lookup dictionary candidates
- `src/apply_dictionary.py`: apply dictionary replacements to detector labeled tokens
- `src/build_llm_prompts.py`: build LLM prompts for LLM candidates.
- `src/run_llm.py`: run LLM inference and merge results.
- `src/evaluate_pipeline.py`: evaluate final pipeline results.
- `src/execute_run_pipeline.py`: execution script: run the 2026 LLM pipeline end to end.
- `src/search_thresholds.py`: run a real detector and entropy threshold grid search with reusable caches.
- `src/plot_threshold_results.py`: plot final ERR and F1 from threshold search results.

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

Set the API key in the shell instead of storing it in the repository:

```bash
export DEEPSEEK_API_KEY="your-api-key"
```

The model is selected with the `MODEL` setting in `src/config.py` or the
`--model` command-line option. Models whose names start with `deepseek-` use
the DeepSeek API; all other model names use the local Ollama API.

## 2. Run

### Check Default Language and Thresholds

Open `config.py`:

```bash
LANGUAGE = os.getenv("PIPELINE_LANGUAGE", "en")
DETECTOR_THRESHOLD = float(os.getenv("PIPELINE_DETECTOR_THRESHOLD", "0.5"))
ENTROPY_THRESHOLD = float(os.getenv("PIPELINE_ENTROPY_THRESHOLD", "0.5"))
MODEL = os.getenv("MODEL", "qwen3.5:9b")

```

### Model Preparation

Prepare detector data and train the detector:

```bash
python -m src.execute_prepare_detector
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

### Grid Search

Run pipeline for threshold grid search (reuse cache including LLM):

```bash
python -m src.search_thresholds \
  --language en \
  --model deepseek-v4-pro
```

This creates:

```text
data/deepseek-v4-pro/en_thresholds/
```

### Plot Grid Search Results

Plot entropy threshold against final ERR and F1, with one line per detector
threshold in each figure:

```bash
python -m src.plot_threshold_results \
  --language en \
  --model deepseek-v4-pro
```

This creates:

```text
data/deepseek-v4-pro/en_thresholds/threshold_entropy_err_en.png
data/deepseek-v4-pro/en_thresholds/threshold_entropy_f1_en.png
```

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
python external/machamp/predict.py \
  models/machamp/detector_en_xlmr_0/model.pt \
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
- `Detector_label`: detector decision, `O` or `NORM`.
- `Detector_confidence`: confidence score of `Detector_label`.
- `Replacement`: current pipeline prediction.
- `Dictionary_entropy`: dictionary entropy if `Source` is `dictionary`, otherwise `null`.
- `Source`: replacement source, such as `keep`, `dictionary`, `llm_pending`, or the LLM model name.
