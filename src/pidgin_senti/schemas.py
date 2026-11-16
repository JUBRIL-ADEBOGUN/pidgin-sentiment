from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


class SentimentLabel(StrEnum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class PredictionRequest(BaseModel):
    text: str = Field(
        ..., min_length=1, max_length=1000, examples=["No wahala, dis app dey work well well 🔥"]
    )
    request_id: str | None = Field(default=None)

    @field_validator("text")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Input text cannot be empty or whitespace only.")
        return cleaned


class PredictionResponse(BaseModel):
    label: SentimentLabel
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: dict[SentimentLabel, float]
    needs_human_review: bool = Field(
        description="True when model confidence falls below SLA threshold."
    )
    latency_ms: float
    model_version: str
