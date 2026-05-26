# LLM Pipeline

Stage 1: train a XLM-R token detector for lexical normalization.

## Data split

```text
weerayut/multilexnorm2026-dev-pub -> Train Split -> TRAIN Data (90%) + DEV Data (10%)
```

- TRAIN Data is used to train the detector and build the lookup dictionary.
- DEV Data is used to evaluate pipeline outputs.

## Files

- `requirements.txt`: dependencies for data preparation.
- `src/inspect_hf_dataset.py`: inspect the HF dataset and export DEV raw/norm data.
- `src/prepare_detector_data.py`: convert HF data to labeled data for training detector 
- `src/dictionary_lookup.py`: build a low-entropy lookup dictionary from TRAIN data.
- `models/machamp/configs/machamp_detector_en.json`: MaChAmp dataset config
- `models/machamp/configs/machamp_params_detector.json`: MaChAmp training config

Label by applying simple rules:
```text
raw token == norm token -> O
raw token != norm token -> NORM
```

## Usage

Download dependencies:

```bash
cd llm_pipeline
conda create -n llm-pipeline python=3.10 -y
conda activate llm-pipeline
pip install -r requirements.txt
```

Prepare training data:

```bash
python -m src.prepare_detector_data
```

This creates data:

```bash
For training: data/machamp/detector_train_en.tsv
For evaluation: data/machamp/detector_dev_en.tsv
```

Build lookup dictionary from TRAIN data:

```bash
python -m src.dictionary_lookup
```

This creates:

```text
data/dictionary_en.jsonl
```

## MaChAmp

Download machamp toolkit:

```bash
mkdir -p external
git clone https://github.com/machamp-nlp/machamp.git external/machamp
pip install -r external/machamp/requirements.txt
```

Check config file before training:

`models/machamp/configs/machamp_detector_en.json`: MaChAmp dataset config
`models/machamp/configs/machamp_params_detector.json`: MaChAmp training config


Train:

```bash
python external/machamp/train.py \
  --dataset_configs models/machamp/configs/machamp_detector_en.json \
  --parameters_config models/machamp/configs/machamp_params_detector.json \
  --model_dir models/machamp/detector_en_xlmr \
  --device 0
```

Predict on the dev split with detector confidence:

```bash
python external/machamp/predict.py \
  models/machamp/detector_en_xlmr_0/model_18.pt \
  data/machamp/detector_dev_en.tsv \
  models/machamp/detector_en_xlmr_0/detector_en.confidence.out \
  --dataset detector_en \
  --device 0 \
  --topn 2
```

Export DEV raw/norm data for evaluation:

```bash
python -m src.inspect_hf_dataset --export-dev-gold
```

This creates:

```text
data/dev_raw_norm_en.jsonl
```

Apply dictionary lookup to detector predictions:

```bash
python -m src.apply_dictionary_to_detector
```

This creates:

```text
data/stage2_dictionary_applied_en.jsonl
data/stage3_llm_candidates_en.jsonl
```
