# src/spd/export_onnx.py
from pathlib import Path

import onnx
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoTokenizer

from spd.categories import NUM_CATEGORIES
from spd.model import MAX_LENGTH, MODEL_NAME, SPDClassifier


def export(
    model_path: str,
    output_dir: str,
    model_name_or_path: str = MODEL_NAME,
) -> str:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = SPDClassifier(num_labels=NUM_CATEGORIES)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()

    dummy = tokenizer(
        "teszt szöveg",
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    fp32_path = out / "model_fp32.onnx"
    # dynamo=True (default in torch>=2.x) is required because the new transformers
    # masking utilities are not TorchScript-traceable. opset_version is set to 18
    # (the minimum supported by the dynamo exporter); onnxruntime 1.18+ fully supports it.
    program = torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"]),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_shapes={
            "input_ids": {0: torch.export.Dim("batch")},
            "attention_mask": {0: torch.export.Dim("batch")},
        },
        opset_version=18,
        dynamo=True,
    )
    program.save(str(fp32_path), external_data=False)

    # The dynamo exporter embeds value_info shape annotations for initializers that
    # conflict with onnxruntime's shape inference during quantization. Strip them first.
    fp32_model = onnx.load(str(fp32_path))
    del fp32_model.graph.value_info[:]
    onnx.save(fp32_model, str(fp32_path))

    int8_path = out / "model.onnx"
    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)
    fp32_path.unlink()

    size_mb = int8_path.stat().st_size / 1e6
    print(f"INT8 model exported to {int8_path}  ({size_mb:.1f} MB)")
    return str(int8_path)


if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else "artifacts/checkpoints/best_model.pt"
    export(model_path, "artifacts")
