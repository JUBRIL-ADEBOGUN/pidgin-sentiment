import re
import time
from collections.abc import Callable
from dataclasses import dataclass

import emoji
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

PIDGIN_MARKERS_RE = re.compile(
    r"\b(?:dey|abeg|wahala|wetin|shey|abi|sha|una|sef|oga|jare|sapa|japa|no be|e choke)\b",
    flags=re.IGNORECASE,
)
NEGATION_RE = re.compile(
    r"\b(?:no|not|never|nor|neither|nothing|nobody|no be|no dey|no fit)\b|n't\b",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class EvalResult:
    metrics: dict[str, float]
    report: str


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error: measures if softmax probabilities match true accuracy."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)

    ece = 0.0
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    for low, high in zip(bin_edges[:-1], bin_edges[1:], strict=True):
        mask = (confidences > low) & (confidences <= high)
        if mask.any():
            bin_acc = accuracies[mask].mean()
            bin_conf = confidences[mask].mean()
            ece += (mask.sum() / len(labels)) * abs(bin_acc - bin_conf)
    return float(ece)


def evaluate_predictions(
    df: pd.DataFrame,
    probs: np.ndarray,
    label_names: list[str],
    prefix: str = "val",
) -> EvalResult:
    y_true = df["label"].to_numpy()
    y_pred = probs.argmax(axis=1)

    metrics: dict[str, float] = {
        f"{prefix}_accuracy": float(accuracy_score(y_true, y_pred)),
        f"{prefix}_macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        f"{prefix}_weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        f"{prefix}_ece": compute_ece(probs, y_true),
    }

    # Per-class F1 (watch Negative F1 closely for escalation/churn monitoring)
    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2])
    for idx, name in enumerate(label_names):
        metrics[f"{prefix}_f1_{name}"] = float(per_class_f1[idx])

    # Pidgin-specific linguistic slices
    texts = df["clean_text"]
    slices = {
        "negation": texts.str.contains(NEGATION_RE, regex=True).to_numpy(),
        "emoji": texts.map(lambda t: emoji.emoji_count(t) > 0).to_numpy(),
        "short_text": (texts.str.split().str.len() < 5).to_numpy(),
        "pidgin_markers": texts.str.contains(PIDGIN_MARKERS_RE, regex=True).to_numpy(),
    }

    for slice_name, mask in slices.items():
        count = int(mask.sum())
        metrics[f"{prefix}_slice_{slice_name}_count"] = float(count)
        if count >= 5:
            metrics[f"{prefix}_slice_{slice_name}_f1"] = float(
                f1_score(y_true[mask], y_pred[mask], average="macro")
            )

    report = classification_report(y_true, y_pred, target_names=label_names, digits=4)
    return EvalResult(metrics=metrics, report=report)


def benchmark_cpu_latency(
    predict_fn: Callable[[list[str]], np.ndarray],
    sample_texts: list[str],
    n_warmup: int = 10,
    n_runs: int = 100,
) -> dict[str, float]:
    """Benchmarks single-request synchronous API latency (batch_size=1)."""
    for i in range(n_warmup):
        predict_fn([sample_texts[i % len(sample_texts)]])

    latencies_ms: list[float] = []
    for i in range(n_runs):
        text = sample_texts[i % len(sample_texts)]
        t0 = time.perf_counter()
        predict_fn([text])
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(latencies_ms)
    return {
        "latency_p50_ms": float(np.percentile(arr, 50)),
        "latency_p95_ms": float(np.percentile(arr, 95)),
        "latency_p99_ms": float(np.percentile(arr, 99)),
    }
