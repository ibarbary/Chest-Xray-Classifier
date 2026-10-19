
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(training: bool):
    operations = []

    # Mild augmentation for training only.
    # Avoid horizontal flips initially because laterality matters in radiology.
    if training:
        operations.append(transforms.RandomRotation(degrees=5))

    operations.extend([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    return transforms.Compose(operations)


class ChestXrayDataset(Dataset):
    def __init__(self, root_dir, split, transform=None):
        self.root_dir = Path(root_dir)
        self.image_dir = self.root_dir / "data" / "raw" / "images-224"
        metadata_path = (
            self.root_dir / "data" / "processed" / "splits.csv"
        )

        df = pd.read_csv(metadata_path)
        self.df = df.loc[df["split"] == split].reset_index(drop=True)

        if self.df.empty:
            raise ValueError(f"No images found for split: {split}")

        self.transform = transform

        # Check paths once when creating the dataset.
        self.image_paths = [
            self.image_dir / filename
            for filename in self.df["Image Index"]
        ]

        missing = [p for p in self.image_paths if not p.is_file()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} image files are missing. "
                f"Example: {missing[0]}"
            )

        self.labels = torch.tensor(
            self.df[LABELS].values,
            dtype=torch.float32,
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image_path = self.image_paths[index]

        # ResNet-18 expects three input channels.
        with Image.open(image_path) as image:
            image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        target = self.labels[index]

        return image, target


def create_dataloaders(root_dir, batch_size=8, num_workers=0):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    datasets = {
        "train": ChestXrayDataset(
            root_dir, "train", get_transforms(training=True)
        ),
        "val": ChestXrayDataset(
            root_dir, "val", get_transforms(training=False)
        ),
        "test": ChestXrayDataset(
            root_dir, "test", get_transforms(training=False)
        ),
    }

    loaders = {}

    for split, dataset in datasets.items():
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )

    return loaders
