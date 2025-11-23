import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


def get_data_loaders(
    train_dir: str, test_dir: str, batch_size: int = 64
) -> tuple[DataLoader, DataLoader]:
    # train transform should have some noise
    train_transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),  # add noise
            transforms.RandomRotation(degrees=15),  # add noise
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # test transform has to stay "clean"
    test_transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = ImageFolder(root=train_dir, transform=train_transform)
    val_dataset = ImageFolder(root=test_dir, transform=test_transform)

    return (
        DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        ),
        DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        ),
    )


if __name__ == "__main__":
    TRAIN_PATH = "../data/raw/train"
    TEST_PATH = "../data/raw/test"

    try:
        train_dl, val_dl = get_data_loaders(TRAIN_PATH, TEST_PATH)

        images, labels = next(iter(train_dl))
        print(f"Success! Batch shape: {images.shape}")
        print(f"Labels: {labels[:5]}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
    except Exception as e:
        print(f"Error loading data: {e}")
