import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.datasets.folder import default_loader


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


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = build_densenet121().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()  # CRITICAL: switch to inference mode

    return model, checkpoint


def build_transform(image_size: int):
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def predict(image_path: Path, model, transform, class_names, device):
    image = default_loader(str(image_path))
    tensor = transform(image).unsqueeze(0).to(device)  # add batch dim -> [1, 3, 224, 224]

    with torch.no_grad():
        probability = model(tensor).item()  # sigmoid output, single float

    predicted_index = int(probability >= 0.5)
    return {
        "image_path": str(image_path),
        "predicted_class": class_names[predicted_index],
        "pneumonia_probability": probability,
        "normal_probability": 1.0 - probability,
    }

def main():
    parser = argparse.ArgumentParser(description="Classify a chest X-ray as NORMAL or PNEUMONIA.")
    parser.add_argument("image", type=Path, help="Path to the X-ray image (.jpeg/.png).")
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("densenet121_best.pth"),
        help="Path to the .pth checkpoint file (default: ./densenet121_best.pth)",
    )
    args = parser.parse_args()

    if not args.image.exists():
        sys.exit(f"Image not found: {args.image}")
    if not args.checkpoint.exists():
        sys.exit(f"Checkpoint not found: {args.checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, checkpoint = load_model(args.checkpoint, device)
    transform = build_transform(checkpoint["image_size"])
    class_names = checkpoint["class_names"]

    print(f"Loaded {checkpoint['model_name']} "
          f"(test ROC-AUC during training: {checkpoint['metrics']['roc_auc']:.4f})")
    print()

    result = predict(args.image, model, transform, class_names, device)

    print(f"Image:      {result['image_path']}")
    print(f"Prediction: {result['predicted_class']}")
    print(f"  P(PNEUMONIA) = {result['pneumonia_probability']:.4f}")
    print(f"  P(NORMAL)    = {result['normal_probability']:.4f}")


if __name__ == "__main__":
    main()