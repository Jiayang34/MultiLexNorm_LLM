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

## 2. Run

### Choose Language

Change language in `src/config.py`

```bash
LANGUAGE = "en"
```

### Model Preparation

Prepare detector data and train the detector:

```bash
python -m src.execute_prepare_detector
```

### Run Pipeline

Run detector prediction, build dictionary, apply dictionary, build LLM prompts, and run LLM:

```bash
python -m src.execute_run_pipeline
```

## 3. Evaluate

Evaluate `overall_final`, `detector`, `dictionary_source`, and `llm_source`:

```bash
python -m src.evaluate_pipeline
```

This creates:

```text
data/en/evaluation_summary_en.json
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
Data for training: data/en/detector_train/detector_train_en.tsv
Data for evaluation: data/en/detector_train/detector_dev_en.tsv
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
  data/en/detector_train/detector_dev_en.tsv \
  data/en/detector_output/detector_en.confidence.out \
  --dataset detector_en \
  --device 0 \
  --topn 2
```

This creates:

```text
data/en/detector_output/detector_en.confidence.out
data/en/detector_output/detector_en.confidence.out.eval
```

4. Build MFR dictionary from TRAIN data:

```bash
python -m src.build_dictionary
```

This creates:

```text
data/en/dictionary_en.jsonl
```

5. Apply dictionary lookup to detector predictions:

```bash
python -m src.apply_dictionary
```

This creates:

```text
data/en/table_applied_dictionary_en.jsonl
data/en/llm_candidates_en.jsonl
```

6. Build LLM prompts for remaining candidates:

```bash
python -m src.build_llm_prompts
```

This creates:

```text
data/en/llm_prompts_en.jsonl
```

7. Run LLM inference

```bash
python -m src.run_llm
```

This creates:

```text
data/en/llm_candidates_applied_llm_en.jsonl
data/en/table_applied_llm_en.jsonl
```

## Master Table

`data/en/table_applied_dictionary_en.jsonl` and
`data/en/table_applied_llm_en.jsonl` use
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
