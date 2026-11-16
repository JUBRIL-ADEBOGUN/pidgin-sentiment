import pytest
from pydantic import ValidationError

from pidgin_senti.schemas import PredictionRequest, PredictionResponse, SentimentLabel


def test_prediction_request_strips_whitespace() -> None:
    req = PredictionRequest(text="   No wahala at all   ")
    assert req.text == "No wahala at all"


def test_prediction_request_rejects_blank_text() -> None:
    with pytest.raises(ValidationError):
        PredictionRequest(text="    \n\t  ")


def test_prediction_response_contract() -> None:
    resp = PredictionResponse(
        label=SentimentLabel.NEGATIVE,
        confidence=0.82,
        probabilities={
            SentimentLabel.POSITIVE: 0.10,
            SentimentLabel.NEUTRAL: 0.08,
            SentimentLabel.NEGATIVE: 0.82,
        },
        needs_human_review=False,
        latency_ms=3.1,
        model_version="baseline-tfidf:baseline.joblib",
    )
    assert resp.label == SentimentLabel.NEGATIVE
    assert resp.needs_human_review is False


def test_pandera_dataset_contract() -> None:
    pd = pytest.importorskip("pandas")
    pa = pytest.importorskip("pandera.pandas")
    from pidgin_senti.data import PidginDatasetSchema

    valid_df = pd.DataFrame(
        {
            "tweet": ["Omo dis food sweet well well"],
            "clean_text": ["Omo dis food sweet well well"],
            "label": [0],
            "label_name": ["positive"],
        }
    )
    validated = PidginDatasetSchema.validate(valid_df)
    assert len(validated) == 1

    invalid_df = pd.DataFrame(
        {
            "tweet": ["Check https://bad.link"],
            "clean_text": ["Check https://bad.link"],  # unmasked URL must fail
            "label": [0],
            "label_name": ["positive"],
        }
    )
    with pytest.raises(pa.errors.SchemaError):
        PidginDatasetSchema.validate(invalid_df)
