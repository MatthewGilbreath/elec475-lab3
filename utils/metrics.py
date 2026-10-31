"""
ELEC 475: Lab 3 - Localization Metrics & Mean-Location Baseline (Starter Template)
Provides:
  1. euclidean_errors: Per-image localization error in original-image pixels.
  2. error_stats: Min, mean, median, max and standard deviation of the errors.
  3. baseline_predictions: Mean-location baseline built from the training labels.
  4. ced_curve: Points of the cumulative error distribution (CED) curve.
  5. best_worst_indices: Indices of the best and worst predictions.
  6. print_results: Accuracy and cost tables (provided).
"""
import os
from typing import Dict, Tuple
import torch
from PIL import Image


def euclidean_errors(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    Computes the Euclidean localization error for every image.

    Args:
        pred : torch.Tensor, shape (N, 2), float. Predicted (u, v) in original-image pixels
        gt   : torch.Tensor, shape (N, 2), float. Ground-truth (u, v) in original-image pixels

    Returns:
        torch.Tensor, shape (N,), float. The Euclidean distance between pred[i] and gt[i],
        in pixels, for each image i
    """
    # ======================================================================
    # [IN-LAB STEP 3]: Implement the Euclidean Error
    #
    # Requirements:
    #   1. For each image, compute the distance between the predicted (u, v)
    #      and the ground-truth (u, v)
    # ======================================================================
    # e_i = sqrt((u_hat - u)^2 + (v_hat - v)^2), done for all N rows at once.
    # diff is (N, 2); squaring is elementwise; sum(dim=1) adds the u and v terms
    # across each row, leaving (N,). Same result as torch.linalg.norm(pred - gt, dim=1).
    diff = pred - gt
    return torch.sqrt((diff ** 2).sum(dim=1))


def error_stats(errors: torch.Tensor) -> Dict[str, float]:
    """
    Summarizes a set of errors.

    Args:
        errors : torch.Tensor, shape (N,), float. Errors in pixels

    Returns:
        dict with exactly the keys "min", "mean", "median", "max" and "std".
        Each value is a Python float (not a tensor).
    """
    # ======================================================================
    # [IN-LAB STEP 3]: Implement the Summary Statistics
    #
    # Requirements:
    #   1. Compute each statistic with errors.min(), .mean(), .median(), .max() and .std()
    #   2. Convert each result to a Python float with .item()
    # ======================================================================
    # Each call returns a 0-D tensor; .item() unwraps it into a plain Python float.
    # Notes: .median() of an even count returns the LOWER of the two middle values
    # (not their average), and .std() uses the sample (N - 1) formula by default.
    return {
        "min": errors.min().item(),
        "mean": errors.mean().item(),
        "median": errors.median().item(),
        "max": errors.max().item(),
        "std": errors.std().item(),
    }


def baseline_predictions(orig_size: torch.Tensor, train_file: str, img_dir: str) -> torch.Tensor:
    """
    The baseline uses the average snout position from the training set as its
    prediction for every test image.

    Args:
        orig_size  : torch.Tensor, shape (N, 2), float. Each test image's size (W, H)
        train_file : path to train_noses.txt
        img_dir    : folder containing the original images

    Returns:
        torch.Tensor, shape (N, 2), float. Baseline prediction (u, v) for each test
        image, in original-image pixels: column 0 is u, column 1 is v
    """
    # ======================================================================
    # [IN-LAB STEP 3]: Implement the Mean-Location Baseline
    #
    # Requirements:
    #   1. For each training image in train_file, compute the normalized snout
    #      position (u / W, v / H): the snout position as a fraction of the image
    #      width and height. Get (W, H) with Image.open(os.path.join(img_dir, name)).size
    #   2. Average the normalized snout positions over the whole training set.
    #      This gives one average normalized snout position, (u_avg, v_avg)
    #   3. Convert the average normalized snout position into a pixel position for
    #      each test image: (u_avg * W, v_avg * H), using that test image's (W, H)
    #      from orig_size. This is the baseline's prediction for that image
    # ======================================================================
    # 1. Normalized snout position of every training image.
    #    Same line format as PetNoseDataset: 'beagle_145.jpg,"(198, 304)"'
    fractions = []
    with open(train_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            name, coords = line.split(",", 1)            # split on the FIRST comma only
            u_str, v_str = coords.strip().strip('"').strip("()").split(",")
            # .size reads only the JPEG header, not the pixels, so this stays fast
            with Image.open(os.path.join(img_dir, name.strip())) as img:
                w, h = img.size                           # PIL gives (W, H)
            fractions.append((float(u_str) / w, float(v_str) / h))

    # 2. One average fraction for the whole training set: (u_avg, v_avg), shape (2,)
    avg = torch.tensor(fractions, dtype=torch.float32).mean(dim=0)

    # 3. Scale to each test image. Broadcasting: (2,) * (N, 2) -> (N, 2), so row i
    #    becomes (u_avg * W_i, v_avg * H_i). Every test image gets the same fraction.
    return avg * orig_size.float()


def ced_curve(errors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the points of the cumulative error distribution (CED) curve.

    Args:
        errors : torch.Tensor, shape (N,), float. Errors in pixels

    Returns:
        tuple (x, y) of two torch.Tensors, each shape (N,), float:
          x : the errors sorted in increasing order, in pixels
          y : y[k-1] = 100 * k / N for k = 1, ..., N, the percentage of predictions
              with an error of at most x[k-1]. The last value is 100
    """
    # ======================================================================
    # [IN-LAB STEP 4]: Implement the CED Curve
    #
    # Hint: use torch.sort
    # ======================================================================
    # x: errors from smallest to largest. torch.sort returns (values, indices);
    # only the values are needed here.
    x, _ = torch.sort(errors)
    # y: the k-th smallest error (k = 1..N) has k of N errors at or below it,
    # so y = 100 * k / N. k starts at 1 (not 0), so the last value is exactly 100.
    n = errors.numel()
    y = 100.0 * torch.arange(1, n + 1, dtype=torch.float32) / n
    return x, y


def best_worst_indices(errors: torch.Tensor, k: int = 4) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Finds the best and worst predictions.

    Args:
        errors : torch.Tensor, shape (N,), float. Errors in pixels
        k      : int. Number of best and worst images to return

    Returns:
        tuple (best, worst) of two torch.Tensors, each shape (k,), int64 (torch.long),
        holding indices into the test set (0 to N-1):
          best  : indices of the k smallest errors, smallest error first
          worst : indices of the k largest errors, largest error first
    """
    # ======================================================================
    # [IN-LAB STEP 5]: Implement the Best and Worst Indices
    #
    # Hint: use torch.argsort
    # ======================================================================
    # argsort returns the INDICES that would sort the errors (not the errors
    # themselves), which is what the plotting code needs to find the images.
    best = torch.argsort(errors)[:k]                     # smallest error first
    worst = torch.argsort(errors, descending=True)[:k]   # largest error first
    return best, worst


def print_results(rows: Dict[str, Dict]):
    """
    Provided. Prints the accuracy and cost tables.

    rows : dict mapping each model name to a dict with the keys
           "errors" (tensor of errors), "params" (int) and "ms" (float)
    """
    stats = {name: error_stats(row["errors"]) for name, row in rows.items()}
    w = max([19] + [len(name) + 2 for name in rows])     # name column width
    print()
    if all(s is not None for s in stats.values()):
        print(f"{'Model':<{w}}" + "".join(f"{h:>8}" for h in ["Min", "Mean", "Median", "Max", "Std"]))
        for name, s in stats.items():
            print(f"{name:<{w}}" + "".join(f"{s[k]:>8.1f}" for k in ["min", "mean", "median", "max", "std"]))
        print()
    else:
        print("[TODO] error_stats() is not implemented yet: accuracy table skipped\n")
    print(f"{'Model':<{w}}{'Params':>13}{'ms/img':>9}")
    for name, row in rows.items():
        print(f"{name:<{w}}{row['params']:>13,}{row['ms']:>9.1f}")
    print()