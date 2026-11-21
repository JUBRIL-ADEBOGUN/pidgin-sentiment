from pathlib import Path

import pandas as pd

from pidgin_senti.predictor import OnnxTransformerPredictor


def inspect_high_confidence_errors(
    split_path: Path = Path("data/processed/validation.parquet"),
    onnx_dir: Path = Path("artifacts/transformer/onnx_int8"),
    top_n: int = 15,
) -> None:
    df = pd.read_parquet(split_path)
    predictor = OnnxTransformerPredictor(onnx_dir=onnx_dir, max_length=80)

    preds = [predictor.predict_one(t) for t in df["clean_text"]]
    df["pred_label"] = [p.label.value for p in preds]
    df["confidence"] = [p.confidence for p in preds]

    errors = df[df["pred_label"] != df["label_name"]].sort_values("confidence", ascending=False)
    print(f"Total errors on {split_path.name}: {len(errors)} / {len(df)}")
    print("\nConfusion Matrix (True x Pred):")
    print(pd.crosstab(df["label_name"], df["pred_label"], margins=True))

    print(f"\nTop {top_n} Highest-Confidence Misclassifications (Loop Back to Stage 1/2):")
    for _, row in errors.head(top_n).iterrows():
        print(
            f"[true={row['label_name']:8s} | pred={row['pred_label']:8s} | "
            f"conf={row['confidence']:.3f}] {row['clean_text']}"
        )

    # Also inspect where true 'neutral' tweets landed so we know if threshold tuning helps
    neutral_df = errors[errors["label_name"] == "neutral"]
    if not neutral_df.empty:
        print(
            f"\nTrue 'neutral' mean confidence on wrong pos/neg pred: "
            f"{neutral_df['confidence'].mean():.3f} (vs overall {df['confidence'].mean():.3f})"
        )


if __name__ == "__main__":
    inspect_high_confidence_errors()
