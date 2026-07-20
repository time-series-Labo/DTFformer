# Long-term forecasting scripts

These scripts reproduce the main comparison protocol for DTFformer and the seven baseline models reported in the paper.

## Models

- DTFformer
- DLinear
- FEDformer
- FilterTS
- PatchTST
- TimeMixer
- WPMixer
- iTransformer

## Shared experimental protocol

To make the comparison transparent, `_run_model.sh` explicitly applies the same training protocol to every model:

| Setting | Value |
| --- | --- |
| Input length | 96 |
| Prediction lengths | 96, 192, 336, 720 |
| Datasets | ETTh1, ETTh2, ETTm1, ETTm2, Weather, Wind1, Wind2 |
| Independent runs | 3 |
| Base seed | 2021; subsequent runs use 2022 and 2023 |
| Encoder layers | 2 |
| Decoder layers | 1 |
| Model dimension | 512 |
| Attention heads | 8 |
| Feed-forward dimension | 2048 |
| Patch length / stride | 16 / 8 |
| Dropout | 0.1 |
| Batch size | 64 |
| Learning rate | 5e-5 |
| Maximum epochs | 10 |
| Early-stopping patience | 3 |
| Loss | MSE |

FilterTS additionally uses the model-specific filtering values published in its official scripts: `quantile=0.9`, `bandwidth=1`, `top_K_static_freqs=10`, and `filter_type=all`. TimeMixer explicitly records its average-pooling down-sampling configuration. These options describe model-specific operations rather than extra training budget.

## Usage

Run one model from the repository root with Bash, Git Bash, or WSL:

```bash
bash scripts/long_term_forecast/DTFformer.sh
bash scripts/long_term_forecast/PatchTST.sh
```

Run the full comparison suite:

```bash
bash scripts/long_term_forecast/run_all.sh
```

The full suite contains 224 experiment settings before accounting for the three independent runs, so it can take a long time. Environment variables can be used for a smaller reproducibility check:

```bash
DTF_DATASETS="ETTh1 weather" \
DTF_PRED_LENS="96 192" \
DTF_ITR=1 \
bash scripts/long_term_forecast/PatchTST.sh
```

Use a specific Python executable when needed:

```bash
PYTHON_BIN="/path/to/python" bash scripts/long_term_forecast/DTFformer.sh
```

Dataset paths and dimensions are selected by `run.py`. Prepare the files according to the data layout in the root README before running these scripts.

## Upstream script references

The baseline command structure and model-specific options were checked against:

- [Time-Series-Library long-term forecasting scripts](https://github.com/thuml/Time-Series-Library/tree/main/scripts/long_term_forecast)
- [FilterTS official scripts](https://github.com/wyl010607/FilterTS/tree/main/scripts)

The files in this repository are adapted to the common protocol reported for DTFformer; they are not verbatim copies of upstream scripts.
