from pathlib import Path

import pandas as pd
import pandera.pandas as pa
from datasets import ClassLabel, DatasetDict, load_dataset
from huggingface_hub import HfApi
from pandera.typing import Series

from pidgin_senti.preprocessing import normalize_pidgin_social_text
from pidgin_senti.schemas import SentimentLabel

LABEL_TO_ID = {
    SentimentLabel.POSITIVE.value: 0,
    SentimentLabel.NEUTRAL.value: 1,
    SentimentLabel.NEGATIVE.value: 2,
}
DEFAULT_ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}

CANDIDATE_REPOS = [
    "HausaNLP/NaijaSenti-Twitter",
    "masakhane/afrisenti",
    "shmuhammad/AfriSenti-twitter-sentiment",
]


class PidginDatasetSchema(pa.DataFrameModel):
    tweet: Series[str] = pa.Field(nullable=False, str_length={"min_value": 1})
    clean_text: Series[str] = pa.Field(nullable=False, str_length={"min_value": 1})
    label: Series[int] = pa.Field(isin=[0, 1, 2])
    label_name: Series[str] = pa.Field(isin=[e.value for e in SentimentLabel])

    @pa.check("clean_text")
    def check_no_raw_urls(cls, s: pd.Series) -> pd.Series:
        return ~s.str.contains(r"https?://", regex=True)


def _load_pcm_from_hub() -> tuple[DatasetDict, str]:
    api = HfApi()
    errors: list[str] = []

    for repo_id in CANDIDATE_REPOS:
        # 1. Try direct config load on refs/convert/parquet
        try:
            ds = load_dataset(repo_id, "pcm", revision="refs/convert/parquet")
            return ds, repo_id
        except Exception as exc:
            errors.append(f"{repo_id} (config): {exc}")

        # 2. Discover exact parquet file paths dynamically from the Hub API
        try:
            files = api.list_repo_files(
                repo_id=repo_id,
                repo_type="dataset",
                revision="refs/convert/parquet",
            )
            pcm_parquets = [f for f in files if f.startswith("pcm/") and f.endswith(".parquet")]
            if not pcm_parquets:
                continue

            data_files = {
                "train": [f"hf://datasets/{repo_id}@refs/convert/parquet/{f}" for f in pcm_parquets if "train" in f],
                "validation": [
                    f"hf://datasets/{repo_id}@refs/convert/parquet/{f}"
                    for f in pcm_parquets
                    if "validation" in f or "dev" in f
                ],
                "test": [f"hf://datasets/{repo_id}@refs/convert/parquet/{f}" for f in pcm_parquets if "test" in f],
            }
            if all(data_files.values()):
                ds = load_dataset("parquet", data_files=data_files)
                return ds, repo_id
        except Exception as exc:
            errors.append(f"{repo_id} (parquet discovery): {exc}")

    raise RuntimeError("Failed to load Nigerian Pidgin (pcm) dataset from all HF mirrors:\n" + "\n".join(errors))


def _normalize_labels(df: pd.DataFrame, label_feature: object) -> pd.DataFrame:
    if pd.api.types.is_integer_dtype(df["label"]):
        names = (
            label_feature.names if isinstance(label_feature, ClassLabel) else [DEFAULT_ID_TO_LABEL[i] for i in range(3)]
        )
        df["label_name"] = df["label"].map(lambda idx: names[int(idx)].strip().lower())
    else:
        df["label_name"] = df["label"].astype(str).str.strip().str.lower()

    # Enforce canonical label IDs: positive=0, neutral=1, negative=2
    df["label"] = df["label_name"].map(LABEL_TO_ID).astype(int)
    return df


def ingest_afrisenti_pcm(output_dir: Path = Path("data/processed")) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ds, source_repo = _load_pcm_from_hub()
    print(f"Loaded Nigerian Pidgin (pcm) dataset from: {source_repo}")

    label_feature = ds["train"].features.get("label")

    for split in ["train", "validation", "test"]:
        df = ds[split].to_pandas()
        df = _normalize_labels(df, label_feature)
        df["clean_text"] = df["tweet"].astype(str).map(normalize_pidgin_social_text)

        # Drop rows that become empty after normalization
        df = df[df["clean_text"].str.len() > 0].copy()

        # Deduplicate training split only to avoid leakage while keeping val/test faithful
        if split == "train":
            before = len(df)
            df = df.drop_duplicates(subset=["clean_text"]).reset_index(drop=True)
            print(f"[train] Dropped {before - len(df)} duplicate normalized tweets.")

        validated_df = PidginDatasetSchema.validate(df)
        out_path = output_dir / f"{split}.parquet"
        validated_df.to_parquet(out_path, index=False)
        print(f"[{split}] Validated & saved {len(validated_df)} rows -> {out_path}")


if __name__ == "__main__":
    ingest_afrisenti_pcm()
