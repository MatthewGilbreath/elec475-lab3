"""
ELEC 475: Lab 3 - Pre-Lab Task 3, Step 2: Dataset Reality Check

1. Iterates every sample of both splits and confirms that each image is
   (3, 227, 227) and each label lies inside the 227 x 227 frame.
2. Plots eight training images with their nose labels drawn as dots and saves
   the figure to plots/dataset_check.png.

Run from the lab folder:
    python -m utils.check_dataset
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save to file only; no GUI window needed
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.dataset import PetNoseDataset, denormalize, IMG_SIZE

BATCH_SIZE = 64
NUM_WORKERS = 4      # set to 0 if DataLoader workers give you trouble
N_PLOT = 8
PLOT_PATH = Path("plots") / "dataset_check.png"


def check_split(split: str) -> bool:
    """Check every sample in one split. Returns True if all samples pass."""
    ds = PetNoseDataset(split=split)
    # shuffle=False keeps batch order == file order, so a bad sample's
    # position in the loop maps straight back to its file name.
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    n_seen, bad_shape, out_of_frame = 0, [], []
    for b, (images, labels, sizes) in enumerate(tqdm(loader, desc=f"{split:5s}")):
        if tuple(images.shape[1:]) != (3, IMG_SIZE, IMG_SIZE):
            bad_shape.append((b, tuple(images.shape)))

        # A label is inside the frame if 0 <= u <= 227 and 0 <= v <= 227.
        inside = ((labels >= 0) & (labels <= IMG_SIZE)).all(dim=1)
        for i in torch.nonzero(~inside).flatten().tolist():
            idx = b * BATCH_SIZE + i
            out_of_frame.append((ds.samples[idx][0], labels[i].tolist()))

        n_seen += images.shape[0]

    print(f"  {split:5s}: {n_seen:,} samples checked | "
          f"bad image shapes: {len(bad_shape)} | labels outside frame: {len(out_of_frame)}")
    for name, lab in out_of_frame[:10]:
        print(f"      outside frame: {name}  label={[round(x, 1) for x in lab]}")
    return not bad_shape and not out_of_frame


def plot_samples(n: int = N_PLOT, seed: int = 0) -> None:
    """Save a grid of n random training images with their nose labels."""
    ds = PetNoseDataset(split="train")
    g = torch.Generator().manual_seed(seed)  # same images every run
    indices = torch.randperm(len(ds), generator=g)[:n].tolist()

    fig, axes = plt.subplots(2, n // 2, figsize=(3 * n // 2, 6.5))
    for ax, idx in zip(axes.flat, indices):
        image, label, _ = ds[idx]
        # Undo normalization (x * STD + MEAN), then (C, H, W) -> (H, W, C) for imshow
        ax.imshow(denormalize(image).permute(1, 2, 0).numpy())
        u, v = label.tolist()
        ax.scatter([u], [v], s=60, c="red", edgecolors="white", linewidths=1.5)
        ax.set_title(ds.samples[idx][0], fontsize=8)
        ax.axis("off")

    fig.suptitle("PetNoseDataset reality check: every dot should sit on a nose")
    fig.tight_layout()
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_PATH, dpi=120)
    plt.close(fig)
    print(f"  Saved {PLOT_PATH}")


if __name__ == "__main__":
    # The __main__ guard is required for num_workers > 0 on macOS and Windows:
    # worker processes re-import this file, and without the guard each one
    # would start its own DataLoader.
    print("Checking every sample:")
    ok = all([check_split("train"), check_split("test")])
    print("  All samples OK" if ok else "  PROBLEMS FOUND (see above)")

    print("Plotting training samples:")
    plot_samples()
