"""
ELEC 475: Lab 3 - Pet Nose Localization Dataset (Starter Template)
Custom PyTorch Dataset for the oxford-iiit-pet-noses re-annotation of the
Oxford-IIIT Pet Dataset.

Each sample returns (image, label, orig_size):
    image     : (3, 227, 227) float tensor, ImageNet-normalized
    label     : (2,) float tensor, nose position (u, v) in the 227x227 image (pixels)
                u = column (x), v = row (y)
    orig_size : (2,) float tensor, original image (W, H)
"""
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

IMG_SIZE = 227

# ImageNet channel statistics: required by the pretrained AlexNet/VGG16 backbones,
# and used by the provided SnoutNet weights as well.
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class PetNoseDataset(Dataset):
    """
    Loads pet images and nose labels, resized to (3, 227, 227).

    Label file format (one sample per line):
        beagle_145.jpg,"(198, 304)"
    The ordered pair is (u, v) in pixels of the ORIGINAL image.
    """
    def __init__(self, root: str = "data/oxford-iiit-pet-noses", split: str = "train"):
        super(PetNoseDataset, self).__init__()
        self.img_dir = Path(root) / "images-original" / "images"
        label_file = Path(root) / f"{split}_noses.txt"

        # ======================================================================
        # [PRE-LAB TASK 3]: Parse the Label File
        #
        # Requirements:
        #   1. Read every non-empty line of label_file
        #   2. Split each line into the image file name and the "(u, v)" string
        #   3. Strip the quotes and brackets, and convert u and v to float
        #   4. Store the results in self.samples as a list of (name, u, v) tuples
        # ======================================================================
        self.samples = []
        with open(label_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # 'beagle_145.jpg,"(198, 304)"' -> 'beagle_145.jpg' and '"(198, 304)"'
                # Split on the FIRST comma only: the (u, v) pair has a comma inside it.
                name, coords = line.split(",", 1)
                # '"(198, 304)"' -> '198, 304'
                coords = coords.strip().strip('"').strip("()")
                u_str, v_str = coords.split(",")
                self.samples.append((name.strip(), float(u_str), float(v_str)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        # ======================================================================
        # [PRE-LAB TASK 3]: Load, Resize and Normalize One Sample
        #
        # Requirements:
        #   1. Open the image with Image.open(path).convert("RGB")
        #      (a few images are grayscale, palette or RGBA)
        #   2. Record the original size: w, h = img.size  (PIL returns (W, H))
        #   3. Resize with img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        #      (the provided weights were trained with exactly this resize)
        #   4. Rescale the label: u * IMG_SIZE / w and v * IMG_SIZE / h
        #   5. Convert with TF.to_tensor, then TF.normalize with MEAN and STD
        #   6. Return (image, label, orig_size) as float32 tensors,
        #      with label = [u, v] and orig_size = [w, h]
        # ======================================================================
        name, u, v = self.samples[idx]

        # 1. Force 3 channels. The `with` block closes the file handle, which
        #    matters when DataLoader workers open thousands of images.
        with Image.open(self.img_dir / name) as im:
            img = im.convert("RGB")

        # 2. Original size BEFORE resizing. PIL gives (W, H), not (H, W).
        w, h = img.size

        # 3. Stretch to 227 x 227 (aspect ratio is NOT preserved).
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)

        # 4. Width and height are stretched by different factors, so u is scaled
        #    by 227/W and v by 227/H.
        label = torch.tensor([u * IMG_SIZE / w, v * IMG_SIZE / h], dtype=torch.float32)
        orig_size = torch.tensor([w, h], dtype=torch.float32)

        # 5. (H, W, C) uint8 [0, 255] -> (C, H, W) float [0, 1] -> ImageNet-normalized
        image = TF.to_tensor(img)
        image = TF.normalize(image, MEAN, STD)

        return image, label, orig_size


def denormalize(img: torch.Tensor) -> torch.Tensor:
    """Undo ImageNet normalization for display (returns values in [0, 1])."""
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    return (img * std + mean).clamp(0, 1)


if __name__ == "__main__":
    print("Testing PetNoseDataset:")
    try:
        for split in ["train", "test"]:
            ds = PetNoseDataset(split=split)
            image, label, orig_size = ds[0]
            print(f"  {split:5s}: {len(ds):,} samples | image {tuple(image.shape)} "
                  f"| label {label.tolist()} | orig_size {orig_size.tolist()}")
    except Exception as e:
        print(f"Dataset not fully implemented yet: {e}")
