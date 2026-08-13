r"""Self-supervised pretraining of the PVT-EDOF encoder on MS-COCO.

The model reconstructs its own grayscale input, so no labels and no
all-in-focus ground truth are needed. The loss is

    L_TOTAL = L1(O, I) + lambda * (1 - MS-SSIM(O, I))

Only the encoder is used afterwards; the decoder exists to make the
reconstruction objective well-posed.

No trained weights ship with this repository. Run this before anything else,
or there is nothing to fuse with::

    python -m pvtedof.train --dataset /path/to/coco/train2017 \
        --epochs 20 --batch-size 6 --embed-dim 96 --learning-rate 1e-5 \
        --ssim-index 2 --save-every 1

See the README for where to get MS-COCO and what to do with the result.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
from PIL import Image
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from .data import IMAGE_EXTENSIONS
from .model import ModelConfig, PVTEDOF
from .third_party import msssim

# lambda in Eq. 5, selected by --ssim-index. The chosen entry is recorded in
# the checkpoint file name (1e0 ... 1e4) so a checkpoint always states the
# weight it was trained with.
SSIM_WEIGHTS = [1, 10, 100, 1000, 10000]
SSIM_TAGS = ["1e0", "1e1", "1e2", "1e3", "1e4"]


class ImageFolderDataset(Dataset):
    """Flat directory of images, loaded as grayscale."""

    def __init__(self, root, transform=None, limit=0):
        self.paths = sorted(
            os.path.join(root, name) for name in os.listdir(root)
            if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS
        )
        if not self.paths:
            raise FileNotFoundError(f"no images found in {root!r}")
        if limit:
            self.paths = self.paths[:limit]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert("L")
        if self.transform:
            image = self.transform(image)
        return image


def timestamp():
    return time.strftime("%Y%m%d_%H%M%S")


def train(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weight = SSIM_WEIGHTS[args.ssim_index]
    tag = SSIM_TAGS[args.ssim_index]
    print(f"device={device}  lambda={weight}  t={args.embed_dim}  "
          f"batch={args.batch_size}  epochs={args.epochs}")

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),  # scales to [0, 1]
    ])
    dataset = ImageFolderDataset(args.dataset, transform=transform,
                                 limit=args.limit)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True)
    print(f"{len(dataset):,} training images from {args.dataset}")

    config = ModelConfig(image_size=args.image_size, embed_dim=args.embed_dim,
                         ape=args.ape)
    model = PVTEDOF(config).to(device)
    stored, executed = model.parameter_counts()
    print(f"model: {stored:,} stored / {executed:,} executed parameters")

    optimizer = Adam(model.parameters(), args.learning_rate)
    l1_loss = torch.nn.L1Loss()

    os.makedirs(args.save_model_dir, exist_ok=True)
    os.makedirs(args.save_loss_dir, exist_ok=True)

    history = {"pixel": [], "ssim": [], "total": []}
    run_id = timestamp()

    model.train()
    for epoch in range(args.epochs):
        with tqdm(loader, unit="batch", ascii=False,
                  desc=f"Epoch {epoch:2d}|{args.epochs - 1}",
                  bar_format="{desc} |{bar:20}| {percentage:3.0f}% "
                             "Batch {n_fmt}|{total_fmt} {postfix}") as progress:
            for img in progress:
                optimizer.zero_grad()
                img = img.to(device)

                outputs = model.autoencoder(img)
                target = img.clone().detach()

                pixel_loss = l1_loss(outputs, target)
                ssim_loss = 1 - msssim(outputs, target, normalize=True)
                total_loss = pixel_loss + weight * ssim_loss

                total_loss.backward()
                optimizer.step()

                history["pixel"].append(pixel_loss.item())
                history["ssim"].append(ssim_loss.item())
                history["total"].append(total_loss.item())
                progress.set_postfix({
                    "Pixel": f"{pixel_loss.item():.4f}",
                    "SSIM": f"{ssim_loss.item():.4f}",
                    "Total": f"{total_loss.item():.4f}",
                })

        if args.save_every and (epoch + 1) % args.save_every == 0:
            path = os.path.join(args.save_model_dir,
                                f"pvtedof_t{args.embed_dim}_{tag}_{run_id}_epoch{epoch:02d}.pt")
            torch.save(model.state_dict(), path)
            print(f"  checkpoint -> {path}")

    for name, values in history.items():
        np.save(os.path.join(args.save_loss_dir,
                             f"loss_{name}_{tag}_{run_id}.npy"), np.array(values))

    model.eval()
    final_path = os.path.join(
        args.save_model_dir,
        f"pvtedof_t{args.embed_dim}_{tag}_{run_id}_final_epoch{args.epochs}.pt")
    torch.save(model.cpu().state_dict(), final_path)
    print(f"\nDone. Final model saved at {final_path}")


def build_parser():
    parser = argparse.ArgumentParser(prog="python -m pvtedof.train",
                                     description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True,
                        help="flat directory of MS-COCO training images")
    parser.add_argument("--limit", type=int, default=0,
                        help="train on only the first N images; 0 uses all of "
                             "them. For the smoke test, not for a real run")
    parser.add_argument("--save-model-dir", default="./checkpoints/")
    parser.add_argument("--save-loss-dir", default="./results/loss/")
    parser.add_argument("--batch-size", type=int, default=6,
                        help="training batch size (6 in the paper)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--ssim-index", type=int, default=2, choices=range(5),
                        help="index into lambda = [1, 10, 100, 1000, 10000]; "
                             "the paper used index 2 (lambda = 100)")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--embed-dim", type=int, default=96,
                        help="number of deep feature maps t (96 in the paper)")
    parser.add_argument("--ape", action="store_true",
                        help="add an absolute position embedding (off in the paper)")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=1,
                        help="save a checkpoint every N epochs; 0 to save only at the end")
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
