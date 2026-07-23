import os

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from torchvision import datasets
from torchvision import transforms
from torchvision import models

from torchvision.models import (
    MobileNet_V2_Weights,
    ResNet18_Weights,
    EfficientNet_B0_Weights
)

from torch.utils.data import DataLoader


# =====================================================
# Настройки
# =====================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

NUM_CLASSES = 10

BATCH_SIZE = 64

EPOCHS = 6

LEARNING_RATE = 1e-3

MODEL_DIR = "saved_models"

DATASET_DIR = "dataset"


# =====================================================
# DataLoader
# =====================================================

def get_dataloaders():

    train_transform = transforms.Compose([

        transforms.Resize((224, 224)),

        transforms.RandomCrop(
            224,
            padding=16
        ),

        transforms.RandomHorizontalFlip(),

        transforms.RandomRotation(15),

        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05
        ),

        transforms.ToTensor(),

        transforms.RandomErasing(
            p=0.25
        ),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    ])

    test_transform = transforms.Compose([

        transforms.Resize((224, 224)),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    ])

    train_dataset = datasets.CIFAR10(

        root=DATASET_DIR,

        train=True,

        download=True,

        transform=train_transform

    )

    test_dataset = datasets.CIFAR10(

        root=DATASET_DIR,

        train=False,

        download=True,

        transform=test_transform

    )

    train_loader = DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=2,

        pin_memory=torch.cuda.is_available()

    )

    test_loader = DataLoader(

        test_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=2,

        pin_memory=torch.cuda.is_available()

    )

    return train_loader, test_loader


# =====================================================
# Заморозка весов
# =====================================================

def freeze_backbone(model):

    for parameter in model.parameters():

        parameter.requires_grad = False

    if hasattr(model, "features"):

        for parameter in model.features[-1].parameters():

            parameter.requires_grad = True

    if hasattr(model, "layer4"):

        for parameter in model.layer4.parameters():

            parameter.requires_grad = True

    if hasattr(model, "classifier"):

        for parameter in model.classifier.parameters():

            parameter.requires_grad = True

    if hasattr(model, "fc"):

        for parameter in model.fc.parameters():

            parameter.requires_grad = True

    return model


# =====================================================
# MobileNetV2
# =====================================================

def create_mobilenet_v2():

    model = models.mobilenet_v2(
        weights=MobileNet_V2_Weights.DEFAULT
    )

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        NUM_CLASSES
    )

    model = freeze_backbone(model)

    model.to(DEVICE)

    return model


# =====================================================
# ResNet18
# =====================================================

def create_resnet18():

    model = models.resnet18(
        weights=ResNet18_Weights.DEFAULT
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        NUM_CLASSES
    )

    model = freeze_backbone(model)

    model.to(DEVICE)

    return model


# =====================================================
# EfficientNet-B0
# =====================================================

def create_efficientnet():

    model = models.efficientnet_b0(
        weights=EfficientNet_B0_Weights.DEFAULT
    )

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        NUM_CLASSES
    )

    model = freeze_backbone(model)

    model.to(DEVICE)

    return model
# =====================================================
# Сохранение модели
# =====================================================

def save_model(model, filename):

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    path = os.path.join(
        MODEL_DIR,
        filename
    )

    torch.save(
        model.state_dict(),
        path
    )


# =====================================================
# Загрузка модели
# =====================================================

def load_model(model, filename):

    path = os.path.join(
        MODEL_DIR,
        filename
    )

    model.load_state_dict(

        torch.load(

            path,

            map_location=DEVICE

        )

    )

    model.to(DEVICE)

    model.eval()

    return model


# =====================================================
# Проверка сохранённых моделей
# =====================================================

def check_saved_models():

    files = [

        "mobilenet_v2.pth",

        "resnet18.pth",

        "efficientnet_b0.pth"

    ]

    for file in files:

        path = os.path.join(

            MODEL_DIR,

            file

        )

        if not os.path.exists(path):

            return False

    return True


# =====================================================
# Обучение модели
# =====================================================

def train_model(model, model_name):

    print(f"\n{'=' * 60}")

    print(f"Обучение {model_name}")

    print("=" * 60)

    train_loader, _ = get_dataloaders()

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(

        filter(

            lambda p: p.requires_grad,

            model.parameters()

        ),

        lr=LEARNING_RATE,

        weight_decay=1e-4

    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(

        optimizer,

        T_max=EPOCHS

    )

    best_acc = 0.0

    best_weights = None

    model.train()

    for epoch in range(EPOCHS):

        running_loss = 0.0

        correct = 0

        total = 0

        progress = tqdm(

            train_loader,

            desc=f"{model_name} | Epoch {epoch + 1}/{EPOCHS}",

            leave=False

        )

        for images, labels in progress:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            _, predicted = outputs.max(1)

            total += labels.size(0)

            correct += predicted.eq(labels).sum().item()

            progress.set_postfix(

                Loss=f"{loss.item():.4f}",

                Acc=f"{100 * correct / total:.2f}%"

            )

        scheduler.step()

        epoch_loss = running_loss / len(train_loader)

        epoch_acc = 100 * correct / total

        if epoch_acc > best_acc:

            best_acc = epoch_acc

            best_weights = {

                k: v.cpu().clone()

                for k, v in model.state_dict().items()

            }

        print(

            f"Epoch {epoch + 1}/{EPOCHS}"

            f" | Loss: {epoch_loss:.4f}"

            f" | Accuracy: {epoch_acc:.2f}%"

        )
    if best_weights is not None:

        model.load_state_dict(best_weights)

    model.eval()

    return model


# =====================================================
# Обучение всех моделей
# =====================================================

def train_all_models():

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    models_dict = {}

    # -----------------------------
    # MobileNetV2
    # -----------------------------

    model = create_mobilenet_v2()

    model = train_model(
        model,
        "MobileNetV2"
    )

    save_model(
        model,
        "mobilenet_v2.pth"
    )

    models_dict["MobileNetV2"] = model

    # -----------------------------
    # ResNet18
    # -----------------------------

    model = create_resnet18()

    model = train_model(
        model,
        "ResNet18"
    )

    save_model(
        model,
        "resnet18.pth"
    )

    models_dict["ResNet18"] = model

    # -----------------------------
    # EfficientNet-B0
    # -----------------------------

    model = create_efficientnet()

    model = train_model(
        model,
        "EfficientNetB0"
    )

    save_model(
        model,
        "efficientnet_b0.pth"
    )

    models_dict["EfficientNetB0"] = model

    print("\nВсе модели успешно обучены.\n")

    return models_dict


# =====================================================
# Загрузка всех моделей
# =====================================================

def load_all_models():

    print("\nЗагрузка моделей...\n")

    models_dict = {}

    model = create_mobilenet_v2()

    models_dict["MobileNetV2"] = load_model(
        model,
        "mobilenet_v2.pth"
    )

    model = create_resnet18()

    models_dict["ResNet18"] = load_model(
        model,
        "resnet18.pth"
    )

    model = create_efficientnet()

    models_dict["EfficientNetB0"] = load_model(
        model,
        "efficientnet_b0.pth"
    )

    print("Все модели загружены.\n")

    return models_dict
# =====================================================
# Test Loader
# =====================================================

def get_test_loader():

    _, test_loader = get_dataloaders()

    return test_loader


# =====================================================
# Calibration Loader
# =====================================================

def get_calibration_loader():

    train_loader, _ = get_dataloaders()

    return train_loader


# =====================================================
# Проверка файла
# =====================================================

if __name__ == "__main__":

    print("=" * 60)
    print("Проверка model_manager.py")
    print("=" * 60)

    print(f"Устройство: {DEVICE}")

    train_loader, test_loader = get_dataloaders()

    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches : {len(test_loader)}")

    models_dict = {

        "MobileNetV2": create_mobilenet_v2(),

        "ResNet18": create_resnet18(),

        "EfficientNetB0": create_efficientnet()

    }

    for name, model in models_dict.items():

        trainable = sum(

            p.numel()

            for p in model.parameters()

            if p.requires_grad

        )

        total = sum(

            p.numel()

            for p in model.parameters()

        )

        print("\n" + "=" * 60)
        print(name)
        print("=" * 60)
        print(f"Trainable parameters : {trainable:,}")
        print(f"Total parameters     : {total:,}")

    print("\nmodel_manager.py готов к работе.")