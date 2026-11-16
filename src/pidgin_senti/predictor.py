from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydantic_settings import BaseSettings, SettingsConfigDict

from pidgin_senti.preprocessing import normalize_pidgin_social_text
from pidgin_senti.schemas import SentimentLabel

LABEL_ORDER = [
    SentimentLabel.POSITIVE,
    SentimentLabel.NEUTRAL,
    SentimentLabel.NEGATIVE,
]


class ServingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PIDGIN_")

    model_backend: str = "auto"  # "auto" | "onnx" | "baseline"
    onnx_dir: Path = Path("artifacts/transformer/onnx_int8")
    onnx_filename: str = "model_quantized.onnx"
    baseline_path: Path = Path("artifacts/baseline/baseline.joblib")
    confidence_threshold: float = 0.60
    max_length: int = 128
    onnx_threads: int = 2


@dataclass(frozen=True)
class RawPrediction:
    label: SentimentLabel
    confidence: float
    probabilities: dict[SentimentLabel, float]
    subword_per_word_ratio: float | None = None


class BasePredictor(ABC):
    version: str

    @abstractmethod
    def predict_one(self, text: str) -> RawPrediction:
        raise NotImplementedError


class BaselineJoblibPredictor(BasePredictor):
    def __init__(self, model_path: Path) -> None:
        import joblib

        self.pipeline = joblib.load(model_path)
        self.version = f"baseline-tfidf:{model_path.name}"

    def predict_one(self, text: str) -> RawPrediction:
        clean = normalize_pidgin_social_text(text)
        probs = self.pipeline.predict_proba([clean])[0]
        return _format_prediction(probs)


class OnnxTransformerPredictor(BasePredictor):
    def __init__(
        self,
        onnx_dir: Path,
        model_filename: str = "model_quantized.onnx",
        max_length: int = 128,
        num_threads: int = 2,
    ) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(str(onnx_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=max_length)
        self.tokenizer.enable_padding(pad_id=1, pad_token="<pad>")

        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = num_threads
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(onnx_dir / model_filename),
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        self.input_names = {inp.name for inp in self.session.get_inputs()}
        self.version = f"afro-xlmr-mini-int8:{model_filename}"

    def predict_one(self, text: str) -> RawPrediction:
        clean = normalize_pidgin_social_text(text)
        enc = self.tokenizer.encode(clean)

        feed = {
            "input_ids": np.array([enc.ids], dtype=np.int64),
            "attention_mask": np.array([enc.attention_mask], dtype=np.int64),
        }
        if "token_type_ids" in self.input_names:
            feed["token_type_ids"] = np.array([enc.type_ids], dtype=np.int64)

        logits = self.session.run(None, feed)[0][0]
        exp = np.exp(logits - np.max(logits))
        probs = exp / exp.sum()

        # Stage 6 linguistic drift proxy: spikes when emerging Pidgin slang fragments subwords
        word_count = max(len(clean.split()), 1)
        non_pad_tokens = sum(enc.attention_mask)
        ratio = non_pad_tokens / word_count

        return _format_prediction(probs, subword_per_word_ratio=ratio)


def _format_prediction(
    probs: np.ndarray,
    subword_per_word_ratio: float | None = None,
) -> RawPrediction:
    best_idx = int(np.argmax(probs))
    prob_map = {label: round(float(probs[i]), 6) for i, label in enumerate(LABEL_ORDER)}
    return RawPrediction(
        label=LABEL_ORDER[best_idx],
        confidence=float(probs[best_idx]),
        probabilities=prob_map,
        subword_per_word_ratio=subword_per_word_ratio,
    )


def load_predictor(settings: ServingSettings | None = None) -> BasePredictor:
    cfg = settings or ServingSettings()
    onnx_file = cfg.onnx_dir / cfg.onnx_filename

    if cfg.model_backend in ("auto", "onnx") and onnx_file.exists():
        return OnnxTransformerPredictor(
            onnx_dir=cfg.onnx_dir,
            model_filename=cfg.onnx_filename,
            max_length=cfg.max_length,
            num_threads=cfg.onnx_threads,
        )

    if cfg.model_backend in ("auto", "baseline") and cfg.baseline_path.exists():
        return BaselineJoblibPredictor(cfg.baseline_path)

    raise FileNotFoundError(
        f"No model artifact found for backend='{cfg.model_backend}'. "
        f"Checked ONNX='{onnx_file}' and Baseline='{cfg.baseline_path}'."
    )
