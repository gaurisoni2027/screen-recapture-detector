"""
train_model.py

Training script for EfficientNet-B0
SalesCode AI Assignment
"""

import copy
import time
from pathlib import Path

import torch
import torch.nn as nn

from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights
)

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.dataset import (
    train_loader,
    val_loader,
    class_names
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

NUM_CLASSES = 2

EPOCHS = 30

LEARNING_RATE = 3e-5

WEIGHT_DECAY = 1e-4

MODEL_PATH = Path("models/efficientnet_best.pth")
MODEL_PATH.parent.mkdir(exist_ok=True)

print("=" * 60)
print("Device :", DEVICE)
print("=" * 60)


def build_model():

    weights = EfficientNet_B0_Weights.DEFAULT

    model = efficientnet_b0(weights=weights)

    # Freeze backbone

    for param in model.features.parameters():
        param.requires_grad = False

    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(

        nn.Dropout(0.4),

        nn.Linear(in_features,512),

        nn.ReLU(),

        nn.Dropout(0.3),

        nn.Linear(512,NUM_CLASSES)

    )

    return model.to(DEVICE)

def get_loss():

    class_counts = torch.tensor([100,82],dtype=torch.float)

    weights = class_counts.sum()/class_counts

    weights = weights/weights.sum()

    weights = weights.to(DEVICE)

    return nn.CrossEntropyLoss(weight=weights)

def accuracy(outputs,labels):

    _,preds = torch.max(outputs,1)

    correct = (preds==labels).sum().item()

    return correct/labels.size(0)

model = build_model()

criterion = get_loss()

optimizer = AdamW(

    model.parameters(),

    lr=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY

)

scheduler = CosineAnnealingLR(

    optimizer,

    T_max=EPOCHS
)

def train_one_epoch():

    model.train()

    running_loss = 0.0

    running_acc = 0.0

    progress = tqdm(train_loader)

    for images, labels in progress:

        images = images.to(DEVICE)

        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        acc = accuracy(outputs, labels)

        running_loss += loss.item()

        running_acc += acc

        progress.set_description(

            f"Train Loss:{loss.item():.4f} Acc:{acc:.4f}"

        )

    return (

        running_loss / len(train_loader),

        running_acc / len(train_loader)

    )

def validate():

    model.eval()

    running_loss = 0.0

    running_acc = 0.0

    predictions = []

    ground_truth = []

    with torch.no_grad():

        progress = tqdm(val_loader)

        for images, labels in progress:

            images = images.to(DEVICE)

            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            acc = accuracy(outputs, labels)

            running_loss += loss.item()

            running_acc += acc

            _, preds = torch.max(outputs,1)

            predictions.extend(preds.cpu().numpy())

            ground_truth.extend(labels.cpu().numpy())

            progress.set_description(

                f"Val Loss:{loss.item():.4f} Acc:{acc:.4f}"

            )

    return (

        running_loss/len(val_loader),

        running_acc/len(val_loader),

        predictions,

        ground_truth

    )

best_accuracy = 0

best_model = copy.deepcopy(model.state_dict())

early_stop = 0

print("\nStarting Training...\n")

for epoch in range(EPOCHS):

    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    if epoch == 5:

        print("\nFine Tuning Started...\n")

        for param in model.features[-2:].parameters():

            param.requires_grad = True

    train_loss, train_acc = train_one_epoch()

    val_loss, val_acc, preds, labels = validate()

    scheduler.step()

    print()

    print(f"Train Loss : {train_loss:.4f}")

    print(f"Train Acc  : {train_acc:.4f}")

    print(f"Val Loss   : {val_loss:.4f}")

    print(f"Val Acc    : {val_acc:.4f}")

    if val_acc > best_accuracy:

        best_accuracy = val_acc

        best_model = copy.deepcopy(model.state_dict())

        torch.save(best_model, MODEL_PATH)

        early_stop = 0

        print("✅ Best Model Saved")

    else:

        early_stop += 1

    if early_stop >= 5:

        print("\nEarly Stopping\n")

        break



print("\nLoading Best Model...\n")

model.load_state_dict(torch.load(MODEL_PATH))

_,_,preds,labels = validate()

print("\nClassification Report\n")

print(

    classification_report(

        labels,

        preds,

        target_names=class_names

    )

)

print("\nConfusion Matrix\n")

print(

    confusion_matrix(

        labels,

        preds

    )

)

print(f"\nBest Validation Accuracy : {best_accuracy:.4f}")

print("\nTraining Complete.")