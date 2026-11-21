import json
from collections import Counter
from pathlib import Path

import numpy as np

CONF_BINS = np.array([0.33, 0.50, 0.60, 0.70, 0.80, 1.01])


def compute_psi(expected: np.ndarray, actual: np.ndarray) -> float:
    """Smoothed 5-bin Population Stability Index (<0.1 stable, >0.25 alert)."""
    exp_counts, _ = np.histogram(expected, bins=CONF_BINS)
    act_counts, _ = np.histogram(actual, bins=CONF_BINS)

    # Laplace smoothing prevents small-sample empty bins from exploding PSI
    exp_pct = (exp_counts + 0.5) / (len(expected) + 0.5 * len(exp_counts))
    act_pct = (act_counts + 0.5) / (len(actual) + 0.5 * len(act_counts))
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def run_drift_check(
    log_path: Path = Path("logs/predictions.jsonl"),
    min_psi_samples: int = 50,
    psi_alert_threshold: float = 0.25,
    frag_alert_threshold: float = 2.2,
) -> dict:
    if not log_path.exists():
        raise FileNotFoundError(f"No inference logs found at {log_path}")

    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line]
    n = len(records)
    if n == 0:
        return {"status": "empty_logs", "sample_count": 0}

    confs = np.array([r["confidence"] for r in records], dtype=float)
    latencies = np.array([r["latency_ms"] for r in records], dtype=float)
    labels = Counter(r["label"] for r in records)
    review_rate = sum(1 for r in records if r["needs_human_review"]) / n

    frag_vals = [r["subword_per_word_ratio"] for r in records if r.get("subword_per_word_ratio") is not None]
    mean_frag = float(np.mean(frag_vals)) if frag_vals else None

    # Reference distribution approximating XLM-T validation confidence profile
    ref_confs = np.clip(np.random.default_rng(42).normal(0.74, 0.11, 1000), 0.34, 1.0)
    conf_psi = compute_psi(ref_confs, confs) if n >= min_psi_samples else None

    alerts = []
    if conf_psi is not None and conf_psi > psi_alert_threshold:
        alerts.append(f"Confidence PSI={conf_psi:.3f} > {psi_alert_threshold}")
    if n >= min_psi_samples and review_rate > 0.30:
        alerts.append(f"High human-review rate={review_rate:.1%}")
    if mean_frag is not None and mean_frag > frag_alert_threshold:
        alerts.append(f"Subword fragmentation={mean_frag:.2f} (emerging Pidgin slang!)")

    status = "INSUFFICIENT_SAMPLES" if n < min_psi_samples else ("ALERT" if alerts else "OK")
    report = {
        "status": status,
        "sample_count": n,
        "label_distribution": {k: round(v / n, 4) for k, v in labels.items()},
        "mean_confidence": round(float(confs.mean()), 4),
        "confidence_psi": round(conf_psi, 4) if conf_psi is not None else None,
        "human_review_rate": round(review_rate, 4),
        "mean_subword_per_word_ratio": round(mean_frag, 3) if mean_frag is not None else None,
        "latency_p50_ms": round(float(np.percentile(latencies, 50)), 2),
        "latency_p99_ms": round(float(np.percentile(latencies, 99)), 2),
        "alerts": alerts,
    }
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    run_drift_check()
