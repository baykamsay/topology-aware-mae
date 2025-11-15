import argparse
import math
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

#!/usr/bin/env python3



def compute_mean_std(root: Path, batch_size: int, workers: int) -> tuple[torch.Tensor, torch.Tensor]:
    dataset = datasets.ImageFolder(root=root, transform=transforms.ToTensor())
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers, pin_memory=True)

    channel_sum = torch.zeros(3)
    channel_sum_sq = torch.zeros(3)
    total_pixels = 0

    for images, _ in loader:
        b, _, h, w = images.shape
        pixels = b * h * w
        total_pixels += pixels
        channel_sum += images.sum(dim=[0, 2, 3])
        channel_sum_sq += (images ** 2).sum(dim=[0, 2, 3])

    mean = channel_sum / total_pixels
    std = torch.sqrt(channel_sum_sq / total_pixels - mean ** 2)
    return mean, std


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute dataset mean/std for ImageNet-like folders.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "pretrain",
        help="Root folder containing ImageNet-style data layout.",
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for iteration.")
    parser.add_argument("--workers", type=int, default=4, help="Number of DataLoader workers.")
    args = parser.parse_args()

    mean, std = compute_mean_std(args.root, args.batch_size, args.workers)
    print(f"Mean: {mean.tolist()}")
    print(f"Std:  {std.tolist()}")


if __name__ == "__main__":
    main()