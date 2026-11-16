from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from pidgin_senti.eval import benchmark_cpu_latency, evaluate_predictions
from pidgin_senti.schemas import SentimentLabel

LABEL_NAMES = [e.value for e in SentimentLabel]


def train_baseline(
    data_dir: Path = Path("data/processed"),
    artifact_dir: Path = Path("artifacts/baseline"),
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    train_df = pd.read_parquet(data_dir / "train.parquet")
    val_df = pd.read_parquet(data_dir / "validation.parquet")
    test_df = pd.read_parquet(data_dir / "test.parquet")

    pipeline = Pipeline(
        steps=[
            (
                "features",
                FeatureUnion(
                    transformer_list=[
                        (
                            "word_tfidf",
                            TfidfVectorizer(
                                ngram_range=(1, 2),
                                min_df=2,
                                max_features=25_000,
                                sublinear_tf=True,
                            ),
                        ),
                        (
                            "char_tfidf",
                            TfidfVectorizer(
                                analyzer="char_wb",
                                ngram_range=(3, 5),
                                min_df=3,
                                max_features=35_000,
                                sublinear_tf=True,
                            ),
                        ),
                    ]
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=1.5,
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    mlflow.set_experiment("pidgin-sentiment")
    with mlflow.start_run(run_name="baseline-tfidf-logreg"):
        pipeline.fit(train_df["clean_text"], train_df["label"])

        val_probs = pipeline.predict_proba(val_df["clean_text"])
        test_probs = pipeline.predict_proba(test_df["clean_text"])

        val_eval = evaluate_predictions(val_df, val_probs, LABEL_NAMES, prefix="val")
        test_eval = evaluate_predictions(test_df, test_probs, LABEL_NAMES, prefix="test")
        latency = benchmark_cpu_latency(pipeline.predict_proba, val_df["clean_text"].tolist()[:50])

        mlflow.log_params({"model_type": "tfidf_word_char_logreg", "C": 1.5})
        mlflow.log_metrics({**val_eval.metrics, **test_eval.metrics, **latency})

        model_path = artifact_dir / "baseline.joblib"
        joblib.dump(pipeline, model_path)
        mlflow.log_artifact(str(model_path))

        print("\n=== Baseline Validation Report ===")
        print(val_eval.report)
        print("Validation & Slice Metrics:", {k: round(v, 4) for k, v in val_eval.metrics.items()})
        print("Test Macro-F1:", round(test_eval.metrics["test_macro_f1"], 4))
        print("CPU Latency (ms):", {k: round(v, 2) for k, v in latency.items()})


if __name__ == "__main__":
    train_baseline()
