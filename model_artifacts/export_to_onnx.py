import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models


def build_densenet121():
    weights_enum = getattr(models, "DenseNet121_Weights", None)
    weights = weights_enum.DEFAULT if weights_enum is not None else None

    try:
        model = models.densenet121(weights=weights)
    except Exception:
        model = models.densenet121(weights=None)

    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 1),
        nn.Sigmoid(),
    )
    return model


def export_onnx(checkpoint_path: Path, output_path: Path, image_size: int, opset: int):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_densenet121()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    dummy_input = torch.randn(1, 3, image_size, image_size, dtype=torch.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["input"],
        output_names=["probability"],
        dynamic_axes={"input": {0: "batch_size"}, "probability": {0: "batch_size"}},
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export DenseNet121 .pth checkpoint to ONNX for web inference.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("model_artifacts/densenet121_best.pth"),
        help="Path to the .pth checkpoint file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("project_website/assets/densenet121_best.onnx"),
        help="Output .onnx path used by the website.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Model input image size.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version.",
    )
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    export_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        image_size=args.image_size,
        opset=args.opset,
    )

    print(f"Export complete: {args.output.resolve()}")
