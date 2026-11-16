import argparse
from pathlib import Path
import json
import mlflow
import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from datasets import Dataset
from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from tokenizers import Tokenizer
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from pidgin_senti.eval import benchmark_cpu_latency, evaluate_predictions
from pidgin_senti.schemas import SentimentLabel

MODEL_ID = "Davlan/afro-xlmr-mini"
LABEL_NAMES = [e.value for e in SentimentLabel]


class WeightedTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


def _softmax(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def _predict_onnx_runtime(
    onnx_dir: Path,
    model_filename: str,
    texts: list[str],
    max_length: int = 128,
    batch_size: int = 32,
) -> tuple[np.ndarray, callable]:
    tokenizer = Tokenizer.from_file(str(onnx_dir / "tokenizer.json"))
    tokenizer.enable_truncation(max_length=max_length)
    tokenizer.enable_padding(pad_id=1, pad_token="<pad>")

    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 2
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(onnx_dir / model_filename),
        sess_options=sess_opts,
        providers=["CPUExecutionProvider"],
    )
    input_names = {inp.name for inp in session.get_inputs()}

    def run_batch(batch_texts: list[str]) -> np.ndarray:
        encodings = tokenizer.encode_batch(batch_texts)
        feed = {
            "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64),
        }
        if "token_type_ids" in input_names:
            feed["token_type_ids"] = np.array([e.type_ids for e in encodings], dtype=np.int64)
        logits = session.run(None, feed)[0]
        return _softmax(logits)

    all_probs = [run_batch(texts[i : i + batch_size]) for i in range(0, len(texts), batch_size)]
    return np.vstack(all_probs), run_batch


def train_and_export(
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 3e-5,
    max_length: int = 128,
    quant_arch: str = "avx2",
    data_dir: Path = Path("data/processed"),
    artifact_dir: Path = Path("artifacts/transformer"),
) -> None:
    pt_dir = artifact_dir / "pytorch_best"
    onnx_fp32_dir = artifact_dir / "onnx_fp32"
    onnx_int8_dir = artifact_dir / "onnx_int8"
    for d in [pt_dir, onnx_fp32_dir, onnx_int8_dir]:
        d.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_parquet(data_dir / "train.parquet")
    val_df = pd.read_parquet(data_dir / "validation.parquet")
    test_df = pd.read_parquet(data_dir / "test.parquet")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    def tokenize_fn(batch):
        return tokenizer(batch["clean_text"], truncation=True, max_length=max_length)

    train_ds = Dataset.from_pandas(train_df[["clean_text", "label"]]).map(tokenize_fn, batched=True)
    val_ds = Dataset.from_pandas(val_df[["clean_text", "label"]]).map(tokenize_fn, batched=True)

    raw_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1, 2]),
        y=train_df["label"].to_numpy(),
    )
    smoothed_weights = np.sqrt(raw_weights)
    class_weights = torch.tensor(smoothed_weights, dtype=torch.float32)

    id2label = {i: name for i, name in enumerate(LABEL_NAMES)}
    label2id = {name: i for i, name in enumerate(LABEL_NAMES)}
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=3, id2label=id2label, label2id=label2id
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return {"macro_f1": float(f1_score(labels, preds, average="macro"))}

    training_args = TrainingArguments(
        output_dir=str(artifact_dir / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        num_train_epochs=epochs,
        weight_decay=0.01,
        warmup_ratio=0.1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        logging_steps=50,
        report_to="none",
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    mlflow.set_experiment("pidgin-sentiment")
    with mlflow.start_run(run_name="afro-xlmr-mini-onnx-int8"):
        trainer.train()
        trainer.save_model(str(pt_dir))
        tokenizer.save_pretrained(str(pt_dir))

        # 1. Export PyTorch -> ONNX FP32
        ort_model = ORTModelForSequenceClassification.from_pretrained(pt_dir, export=True)
        ort_model.save_pretrained(onnx_fp32_dir)
        tokenizer.save_pretrained(onnx_fp32_dir)

        # 2. Quantize ONNX FP32 -> ONNX INT8
        qconfig_fn = getattr(AutoQuantizationConfig, quant_arch)
        qconfig = qconfig_fn(is_static=False, per_channel=False)
        quantizer = ORTQuantizer.from_pretrained(onnx_fp32_dir)
        quantizer.quantize(save_dir=onnx_int8_dir, quantization_config=qconfig)
        tokenizer.save_pretrained(onnx_int8_dir)

        # 3. Verify FP32 vs INT8 parity + Benchmark pure ONNX Runtime CPU Latency
        fp32_val_probs, _ = _predict_onnx_runtime(
            onnx_fp32_dir, "model.onnx", val_df["clean_text"].tolist(), max_length=max_length
        )
        int8_val_probs, int8_infer_fn = _predict_onnx_runtime(
            onnx_int8_dir,
            "model_quantized.onnx",
            val_df["clean_text"].tolist(),
            max_length=max_length,
        )
        int8_test_probs, _ = _predict_onnx_runtime(
            onnx_int8_dir,
            "model_quantized.onnx",
            test_df["clean_text"].tolist(),
            max_length=max_length,
        )

        fp32_eval = evaluate_predictions(val_df, fp32_val_probs, LABEL_NAMES, prefix="fp32_val")
        int8_val_eval = evaluate_predictions(val_df, int8_val_probs, LABEL_NAMES, prefix="int8_val")
        int8_test_eval = evaluate_predictions(
            test_df, int8_test_probs, LABEL_NAMES, prefix="int8_test"
        )
        latency = benchmark_cpu_latency(int8_infer_fn, val_df["clean_text"].tolist()[:50])

        mlflow.log_params(
            {
                "base_model": MODEL_ID,
                "epochs": epochs,
                "lr": lr,
                "max_length": max_length,
                "quant_arch": quant_arch,
            }
        )
        mlflow.log_metrics(
            {
                **fp32_eval.metrics,
                **int8_val_eval.metrics,
                **int8_test_eval.metrics,
                **latency,
            }
        )

        summary = {
            "baseline_comparison_targets": {
                "val_macro_f1": 0.4633,
                "test_macro_f1": 0.4668,
                "val_f1_neutral": 0.0,
                "val_slice_negation_f1": 0.4577,
                "cpu_latency_p99_ms": 5.02,
            },
            "transformer_onnx_int8": {
                "fp32_val_macro_f1": fp32_eval.metrics["fp32_val_macro_f1"],
                **int8_val_eval.metrics,
                **int8_test_eval.metrics,
                **latency,
            },
        }
        (onnx_int8_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2))
        (onnx_int8_dir / "eval_report.txt").write_text(
            f"=== Validation Report ===\n{int8_val_eval.report}\n"
            f"=== Test Report ===\n{int8_test_eval.report}\n"
        )

        print("\n=== ONNX INT8 Test Report ===")
        print(int8_test_eval.report)
        print(f"FP32 Val Macro-F1: {fp32_eval.metrics['fp32_val_macro_f1']:.4f}")
        print(f"INT8 Val Macro-F1: {int8_val_eval.metrics['int8_val_macro_f1']:.4f}")
        print(f"INT8 Test Macro-F1: {int8_test_eval.metrics['int8_test_macro_f1']:.4f}")
        print("INT8 CPU Latency (ms):", {k: round(v, 2) for k, v in latency.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--quant-arch", type=str, default="avx2", choices=["avx2", "avx512", "arm64"]
    )
    args = parser.parse_args()
    train_and_export(epochs=args.epochs, batch_size=args.batch_size, quant_arch=args.quant_arch)
