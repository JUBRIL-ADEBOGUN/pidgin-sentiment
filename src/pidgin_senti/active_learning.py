import csv
import json
from pathlib import Path


def export_active_learning_batch(
    log_path: Path = Path("logs/predictions.jsonl"),
    out_csv: Path = Path("data/review_queue/to_label.csv"),
    top_k: int = 100,
) -> None:
    if not log_path.exists():
        raise FileNotFoundError(f"No logs found at {log_path}")

    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line]
    seen: set[str] = set()
    scored: list[tuple[float, dict]] = []

    for r in records:
        text = r["clean_text"]
        if text in seen:
            continue
        seen.add(text)

        probs = r["probabilities"]
        # High acquisition score when positive & negative tie (candidate neutral/sarcasm)
        pos_neg_gap = abs(probs["positive"] - probs["negative"])
        uncertainty = 1.0 - r["confidence"]
        slang_bonus = 0.25 if (r.get("subword_per_word_ratio") or 1.0) > 2.0 else 0.0
        priority = uncertainty + (1.0 - pos_neg_gap) + slang_bonus
        scored.append((priority, r))

    scored.sort(key=lambda x: x[0], reverse=True)
    selected = scored[:top_k]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["clean_text", "predicted_label", "confidence", "priority", "human_label"])
        for score, r in selected:
            writer.writerow([r["clean_text"], r["label"], r["confidence"], round(score, 4), ""])

    print(f"Exported {len(selected)} high-value Pidgin samples for annotation -> {out_csv}")


if __name__ == "__main__":
    export_active_learning_batch()
