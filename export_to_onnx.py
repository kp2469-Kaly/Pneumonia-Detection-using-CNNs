"""
Export densenet121_best.pth to ONNX format for in-browser inference.

Usage:
    python export_to_onnx.py

Output:
    project_website/densenet121_best.onnx  (~30 MB)

After running this script, push the ONNX file to GitHub along with the
website files. The 'Try It Live' section will load and run the model
entirely in the visitor's browser using ONNX Runtime Web (no server needed).
"""

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models


def build_densenet121() -> nn.Module:
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


def main() -> None:
    checkpoint_path = Path("model_artifacts/densenet121_best.pth")
    output_path = Path("project_website/densenet121_best.onnx")

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Run this script from the project root directory."
        )

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model = build_densenet121()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # Dummy input matching the expected preprocessing output
    dummy_input = torch.zeros(1, 3, 224, 224, dtype=torch.float32)

    print(f"Exporting to ONNX: {output_path}")
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=11,
    )

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Done.  File size: {size_mb:.1f} MB  →  {output_path}")
    print()
    print("Next steps:")
    print("  1. Commit project_website/densenet121_best.onnx to your GitHub repo.")
    print("  2. Enable GitHub Pages (Settings → Pages → Deploy from branch).")
    print("  3. Open the site and use the 'Try It Live' section to upload an X-ray.")


if __name__ == "__main__":
    main()
