# Pidgin Sentiment

An end-to-end machine-learning project for three-class sentiment analysis of
Nigerian Pidgin (`pcm`) social-media text. The project includes dataset
ingestion and validation, a TF-IDF baseline, an AfroXLMR-mini transformer
pipeline, ONNX INT8 export, evaluation, and a FastAPI inference service.

## What the pipeline does

1. **Ingest and validate data.** Download a Pidgin split from an available
   Hugging Face dataset mirror, normalize labels and text, validate rows with
   Pandera, and write Parquet splits.
2. **Train candidate models.** Fit a word/character TF-IDF + Logistic
   Regression baseline or fine-tune `Davlan/afro-xlmr-mini`.
3. **Evaluate.** Report overall and per-class performance, calibration,
   Pidgin-oriented linguistic slices, and CPU inference latency. Log training
   parameters and metrics to MLflow.
4. **Export and serve.** Export the transformer to ONNX FP32 and dynamically
   quantized INT8, then serve ONNX or baseline predictions through FastAPI.

The canonical label IDs used by the training and data pipeline are:

| ID | Label |
|---:|---|
| 0 | positive |
| 1 | neutral |
| 2 | negative |

## Repository layout

```text
.
├── .github/workflows/
│   ├── ci.yml                 # Ruff and unit/contract tests
│   └── train.yml              # Manually triggered transformer training/export
├── .vscode/                   # Recommended editor settings and extensions
├── data/processed/            # Validated Parquet splits (generated; not committed)
├── artifacts/                 # Trained models and evaluation outputs (generated)
├── src/pidgin_senti/
│   ├── api.py                 # FastAPI routes and Prometheus metrics
│   ├── data.py                # Hugging Face ingestion and Pandera schema
│   ├── eval.py                # Metrics, linguistic slices, latency benchmark
│   ├── predictor.py           # Predictor interfaces and backend loading
│   ├── preprocessing.py       # Shared social-text normalization
│   ├── schemas.py             # API request/response and label contracts
│   ├── train_baseline.py      # TF-IDF + Logistic Regression
│   └── train_transformer.py   # AfroXLMR-mini fine-tuning and ONNX export
├── tests/
├── Dockerfile
└── pyproject.toml
```

Generated datasets, model artifacts, MLflow runs, and local environment files
are excluded from version control. Review the project `.gitignore` before
adding any data or model files.

## Requirements and setup

- Python 3.11
- [uv](https://docs.astral.sh/uv/) for environment and lockfile management
- Docker for building the serving image
- Network access to Hugging Face for dataset/model downloads
- A CPU is sufficient for inference and the baseline. A CUDA-capable GPU can
  accelerate transformer training.

Install the complete development and training environment:

```bash
uv sync --extra train --extra serve --extra serve-baseline --extra dev
```

For only unit tests and linting:

```bash
uv sync --extra dev
```

The dependency groups are defined in `pyproject.toml`; `uv.lock` records the
resolved environment. Use `uv run` for commands below so they run in the
project environment.

## Data ingestion

Download and validate the Nigerian Pidgin dataset:

```bash
uv run python -m pidgin_senti.data
```

The ingestion code tries the `pcm` configuration on these Hugging Face dataset
mirrors: `HausaNLP/NaijaSenti-Twitter`, `masakhane/afrisenti`, and
`shmuhammad/AfriSenti-twitter-sentiment`. On success it:

- standardizes sentiment IDs and label names;
- applies the shared normalizer, which masks URLs and mentions, fixes
  malformed Unicode, and preserves emoji by default;
- drops empty normalized rows;
- removes duplicate normalized text from the training split only; and
- validates each split against the Pandera contract before writing:

```text
data/processed/
├── train.parquet
├── validation.parquet
└── test.parquet
```

If ingestion cannot find a valid source or a required split, it raises an
error containing the attempted mirror details rather than silently producing
partial data.

## Training

### TF-IDF baseline

Train the Logistic Regression baseline from the processed Parquet splits:

```bash
uv run python -m pidgin_senti.train_baseline
```

The baseline combines word and character TF-IDF features. It writes
`artifacts/baseline/baseline.joblib`, evaluates validation and test data, and
logs its parameters and metrics to MLflow.

### AfroXLMR-mini transformer

Fine-tune the default `Davlan/afro-xlmr-mini` model, export ONNX FP32 and
quantized INT8 versions, and evaluate both:

```bash
uv run python -m pidgin_senti.train_transformer
```

The training command accepts:

```text
--model-id       Hugging Face model ID (default: Davlan/afro-xlmr-mini)
--epochs         Number of training epochs (default: 4)
--batch-size     Training batch size (default: 8)
--lr             Learning rate (default: 5e-5)
--max-length     Tokenizer truncation length (default: 96)
--quant-arch     Quantization target: avx2, avx512, or arm64 (default: avx2)
```

For example:

```bash
uv run python -m pidgin_senti.train_transformer \
  --epochs 3 \
  --batch-size 8 \
  --max-length 96 \
  --quant-arch avx2
```

The transformer pipeline uses class-weighted loss, early stopping based on
validation macro-F1, and ONNX dynamic quantization. It evaluates validation and
test splits and benchmarks single-item CPU inference. Outputs are written
under `artifacts/transformer/`, including:

```text
artifacts/transformer/
├── pytorch_best/
├── onnx_fp32/
└── onnx_int8/
    ├── model_quantized.onnx
    ├── tokenizer.json
    ├── metrics_summary.json
    └── eval_report.txt
```

MLflow uses the local tracking directory by default. To use a tracking server,
set `MLFLOW_TRACKING_URI` before launching training. Do not copy private text
or credentials into experiment parameters or logs.

## Evaluation

Evaluation is implemented in `pidgin_senti.eval`. It reports accuracy,
macro-F1, weighted-F1, expected calibration error, and per-class F1. It also
measures Pidgin-oriented slices:

- negation markers;
- emoji-containing text;
- short text;
- common Pidgin markers.

Slice F1 is reported only when a slice contains at least five examples.
Latency benchmarking warms up the predictor and reports P50, P95, and P99
single-request latency on the machine running the command. These are local
measurements, not a production latency guarantee.

## Run the API locally

After generating either a baseline or ONNX model, start the service:

```bash
uv run uvicorn pidgin_senti.api:app --host 0.0.0.0 --port 8000
```

The service selects a backend using `PIDGIN_MODEL_BACKEND`:

| Variable | Default | Description |
|---|---|---|
| `PIDGIN_MODEL_BACKEND` | `auto` | `auto`, `onnx`, or `baseline`; `auto` prefers ONNX when its artifact exists |
| `PIDGIN_ONNX_DIR` | `artifacts/transformer/onnx_int8` | ONNX model directory |
| `PIDGIN_ONNX_FILENAME` | `model_quantized.onnx` | ONNX graph filename |
| `PIDGIN_BASELINE_PATH` | `artifacts/baseline/baseline.joblib` | Baseline model file |
| `PIDGIN_CONFIDENCE_THRESHOLD` | `0.60` | Predictions below this confidence are flagged for human review |
| `PIDGIN_MAX_LENGTH` | `128` | Maximum ONNX tokenizer sequence length |
| `PIDGIN_ONNX_THREADS` | `2` | ONNX Runtime intra-op CPU threads |

For example, to force baseline inference:

```bash
PIDGIN_MODEL_BACKEND=baseline \
  uv run uvicorn pidgin_senti.api:app --host 0.0.0.0 --port 8000
```

### HTTP endpoints

| Method and path | Purpose |
|---|---|
| `GET /health` | Liveness check |
| `GET /ready` | Readiness check with loaded model version |
| `POST /predict` | Sentiment prediction |
| `GET /metrics` | Prometheus metrics |

Example request:

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"No wahala, dis app dey work well well 🔥"}'
```

The response contains the predicted label, confidence, probabilities for all
three labels, human-review flag, latency in milliseconds, and model version.
Invalid or whitespace-only text is rejected by request validation.

### Monitoring

The `/metrics` endpoint exports prediction counts by label/review status,
inference latency, prediction confidence, and the tokenizer subword-per-word
ratio. The latter is a proxy for vocabulary drift as new slang becomes more
fragmented; it is not a direct measurement of unknown words. Raw request text
is not included in these metrics.

## Docker

The image installs serving dependencies and copies `artifacts/` into the
runtime image. Generate at least one supported model artifact before building:

```bash
uv run python -m pidgin_senti.data
uv run python -m pidgin_senti.train_baseline
# Or train/export the transformer instead:
# uv run python -m pidgin_senti.train_transformer

docker build -t pidgin-senti:local .
docker run --rm -p 8000:8000 pidgin-senti:local
```

To select a backend or custom artifact location at runtime, pass the
corresponding `PIDGIN_*` environment variables to `docker run`.

## Development and CI

Run the unit and contract tests and Ruff checks locally:

```bash
uv run pytest -v
uv run ruff check .
uv run ruff format --check .
```

The GitHub Actions CI workflow runs Ruff and tests on pushes and pull requests
targeting `main`. The separate training workflow is manually triggered; it
ingests the data, trains and quantizes the transformer, publishes evaluation
results to the workflow summary, and uploads the ONNX INT8 artifacts.

## Responsible use and limitations

This model is an experimental classification tool, not a source of verified
facts or a substitute for human judgment. Performance may vary across dialects,
code-switching, spelling conventions, platforms, and emerging expressions.
Review class- and slice-level evaluation before deployment, set an appropriate
human-review threshold, and monitor confidence and subword-fragmentation
metrics after release. Protect any downloaded data and generated artifacts
according to their applicable licenses and access requirements.
