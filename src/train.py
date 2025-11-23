import os
import time

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import get_data_loaders
from src.model import FaceModel


def train_model(
    data_dir: str,
    save_dir: str = "../data/models",
    num_epochs: int = 25,
    batch_size: int = 64,
    learning_rate: float = 0.001,
) -> dict[str, list]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")
    train_loader, test_loader = get_data_loaders(
        train_dir, test_dir, batch_size=batch_size
    )

    model = FaceModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.1, patience=5
    )

    best_acc = 0.0

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    start_time = time.time()

    for epoch in range(num_epochs):
        print(f"---Epoch: {epoch + 1} / {num_epochs}---")
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        scheduler.step(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"{save_dir}/best_model.pth")
            print(f"Saved new best model with acc: {val_acc:.4f}")

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time // 60:.0f}m {total_time % 60:.0f}s")
    print(f"Best Validation Accuracy: {best_acc:.4f}")

    return history


def train_one_epoch(
    model: FaceModel,
    loader: DataLoader,
    loss_criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_corrects = 0
    loop = tqdm(loader)

    for image_batch, labels in loop:
        image_batch = image_batch.to(device)
        labels = labels.to(device)

        outputs = model(image_batch)
        loss = loss_criterion(outputs, labels)

        # Backward Pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * image_batch.size(0)

        # preds = list of bool values
        _, preds = torch.max(outputs, 1)
        running_corrects += (preds == labels).sum().item()

        # progress bar
        loop.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = running_corrects / len(loader.dataset)

    return epoch_loss, epoch_acc


def evaluate(
    model: FaceModel,
    loader: DataLoader,
    loss_criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    loop = tqdm(loader)

    # faster with no grad
    with torch.no_grad():
        for image_batch, labels in loop:
            image_batch = image_batch.to(device)
            labels = labels.to(device)

            outputs = model(image_batch)
            loss = loss_criterion(outputs, labels)

            running_loss += loss.item() * image_batch.size(0)

            # preds = list of bool values
            _, preds = torch.max(outputs, 1)
            running_corrects += (preds == labels).sum().item()

            # progress bar
            loop.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = running_corrects / len(loader.dataset)

    return epoch_loss, epoch_acc


def main():
    mock_data = []
    for _ in range(10):
        img = torch.randn(3, 224, 224)  # random noise
        label = torch.randint(0, 7, (1,)).item()  # random label 0-6
        mock_data.append((img, label))

    mock_loader = DataLoader(mock_data, batch_size=2, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FaceModel(num_classes=7).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print("\n--- Testing Training Step ---")
    try:
        t_loss, t_acc = train_one_epoch(
            model, mock_loader, criterion, optimizer, device
        )
        print(f"Train Loop Success! Loss: {t_loss:.4f}, Acc: {t_acc:.4f}")
    except Exception as e:
        print(f"Train Loop Failed: {e}")

    print("\n--- Testing Evaluation Step ---")
    try:
        e_loss, e_acc = evaluate(model, mock_loader, criterion, device)
        print(f"Eval Loop Success! Loss: {e_loss:.4f}, Acc: {e_acc:.4f}")
    except Exception as e:
        print(f"Eval Loop Failed: {e}")


if __name__ == "__main__":
    DATA_DIR = "../data/raw/"
    train_model(data_dir=DATA_DIR, num_epochs=25)
    # main()
