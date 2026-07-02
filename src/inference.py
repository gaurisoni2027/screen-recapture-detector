"""
inference.py

Run inference on a single image using the trained
EfficientNet model.
"""

from pathlib import Path
import time

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "models/efficientnet_best.pth"

CLASS_NAMES = ["real", "screen"]

transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_model():

    model = efficientnet_b0(weights=None)

    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 2)
    )

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )
    )

    model.to(DEVICE)

    model.eval()

    return model


model = load_model()


def predict(image_path):

    image = Image.open(image_path).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0).to(DEVICE)

    start = time.time()

    with torch.no_grad():

        output = model(image)

        probabilities = torch.softmax(output, dim=1)[0]

        screen_probability = probabilities[1].item()

        prediction = torch.argmax(probabilities).item()

    end = time.time()

    return (
        CLASS_NAMES[prediction],
        screen_probability,
        (end - start) * 1000
    )


if __name__ == "__main__":

    image = "data/raw/real/flower.jpg"

    label, probability, inference = predict(image)

    print("=" * 50)
    print("Prediction :", label.upper())
    print(f"Screen Probability : {probability:.4f}")
    print(f"Inference : {inference:.2f} ms")
    print("=" * 50)