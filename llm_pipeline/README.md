# LLM Pipeline

Stage 1: train a token detector for lexical normalization

## Files

- `requirements.txt`: dependencies for data preparation.
- `src/prepare_detector_data.py`: convert HF data to labeled data for training detector 
- `src/debug_prepare_data.py`: prints sample raw/norm/label for debugging
- `configs/machamp_detector_en.json`: MaChAmp dataset config
- `configs/machamp_params_detector.json`: MaChAmp training config

Label by applying simple rules:
```text
raw token == norm token -> O
raw token != norm token -> NORM
```

## Usage

Run from this folder:

```bash
cd llm_pipeline
pip install -r requirements.txt
python -m src.prepare_detector_data
python -m src.debug_prepare_data
```

This creates:

```text
For debugging:
data/detector_train_en.jsonl

For training detector:
data/machamp/detector_train_en.tsv
data/machamp/detector_dev_en.tsv
```

## MaChAmp

Download machamp toolkit:

```bash
mkdir -p external
git clone https://github.com/machamp-nlp/machamp.git external/machamp
pip install -r external/machamp/requirements.txt
```

Check config file before training:

```bash
`configs/machamp_detector_en.json`: MaChAmp dataset config
`configs/machamp_params_detector.json`: MaChAmp training config
```

Train:

```bash
python external/machamp/train.py \
  --dataset_configs configs/machamp_detector_en.json \
  --parameters_config configs/machamp_params_detector.json \
  --model_dir models/machamp/detector_en_xlmr \
  --device 0
```
