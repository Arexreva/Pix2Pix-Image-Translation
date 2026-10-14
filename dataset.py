from PIL import Image
import numpy as np
import os
from torch.utils.data import Dataset
import config


class MapDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir

        self.list_files = sorted([
            f for f in os.listdir(root_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        print(f"Found {len(self.list_files)} images in {root_dir}")

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)

        image = np.array(
            Image.open(img_path).convert("RGB")
        )

        width = image.shape[1]
        input_image = image[:, :width // 2, :]
        target_image = image[:, width // 2:, :]

        augmentations = config.both_transform(
            image=input_image,
            image0=target_image
        )

        input_image = augmentations["image"]
        target_image = augmentations["image0"]

        input_image = config.transform_only_input(
            image=input_image
        )["image"]

        target_image = config.transform_only_input(
            image=target_image
        )["image"]

        return input_image, target_image
