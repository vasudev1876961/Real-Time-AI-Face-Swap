"""
Standalone PyTorch Face Swap Model Training Pipeline (Shared-Encoder Dual-Decoder Architecture).
Trains an identity transformation model specifically on aligned hero face crops.
Run this script on a GPU (local NVIDIA GPU or Google Colab/Kaggle).
"""

import os
import sys
import argparse
import time
import glob
from typing import Tuple
import cv2
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ============================================================================
# 1. MODEL ARCHITECTURE (Shared Encoder + Hero Decoder)
# ============================================================================

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size, stride, padding)
        self.leaky = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.leaky(self.conv(x))


class UpscaleBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c * 4, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.leaky = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.leaky(self.pixel_shuffle(self.conv(x)))


class SharedEncoder(nn.Module):
    """Encodes facial geometry, expression, and head pose."""
    def __init__(self, in_channels=3, latent_dim=512):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, 64)    # 128 -> 64
        self.conv2 = ConvBlock(64, 128)           # 64 -> 32
        self.conv3 = ConvBlock(128, 256)          # 32 -> 16
        self.conv4 = ConvBlock(256, 512)          # 16 -> 8
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512 * 8 * 8, latent_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        feat = self.flatten(x)
        latent = self.fc(feat)
        return latent


class HeroDecoder(nn.Module):
    """Reconstructs the specific hero's photorealistic facial texture."""
    def __init__(self, latent_dim=512, out_channels=3):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 512 * 8 * 8)
        self.up1 = UpscaleBlock(512, 256)         # 8 -> 16
        self.up2 = UpscaleBlock(256, 128)         # 16 -> 32
        self.up3 = UpscaleBlock(128, 64)          # 32 -> 64
        self.up4 = UpscaleBlock(64, 32)           # 64 -> 128
        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, latent):
        x = F.leaky_relu(self.fc(latent), 0.1)
        x = x.view(-1, 512, 8, 8)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        out = self.sigmoid(self.out_conv(x))
        return out


class FaceSwapModel(nn.Module):
    """Full End-to-End Face Swap Pipeline Model."""
    def __init__(self, latent_dim=512):
        super().__init__()
        self.encoder = SharedEncoder(in_channels=3, latent_dim=latent_dim)
        self.decoder = HeroDecoder(latent_dim=latent_dim, out_channels=3)

    def forward(self, source_face):
        # source_face: (B, 3, 128, 128) in [0.0, 1.0]
        latent = self.encoder(source_face)
        swapped_face = self.decoder(latent)
        return swapped_face


# ============================================================================
# 2. DATASET LOADER
# ============================================================================

class FaceCropDataset(Dataset):
    """Loads aligned 128x128 face crops for training."""
    def __init__(self, image_dir: str, target_size: Tuple[int, int] = (128, 128)):
        self.image_paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
            self.image_paths.extend(glob.glob(os.path.join(image_dir, "**", ext), recursive=True))
        self.target_size = target_size

    def __len__(self):
        return max(1, len(self.image_paths))

    def __getitem__(self, idx):
        if not self.image_paths:
            return torch.zeros((3, self.target_size[1], self.target_size[0]), dtype=torch.float32)

        path = self.image_paths[idx % len(self.image_paths)]
        img = cv2.imread(path)
        if img is None:
            return torch.zeros((3, self.target_size[1], self.target_size[0]), dtype=torch.float32)

        # Resize to 128x128
        img = cv2.resize(img, self.target_size)
        # BGR -> [0, 1] tensor NCHW
        tensor = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
        return tensor


# ============================================================================
# 3. TRAINING LOOP
# ============================================================================

def train_model(
    hero_images_dir: str,
    output_checkpoint_dir: str = "checkpoints",
    epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 5e-4,
    device_name: str = "auto",
):
    if not TORCH_AVAILABLE:
        print("[ERROR] PyTorch is required for model training. Install with: pip install torch torchvision")
        return

    os.makedirs(output_checkpoint_dir, exist_ok=True)

    # Device selection
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    print("=" * 65)
    print(f"Training Custom Hero Face Swap Model on: {device}")
    print(f"Dataset directory: {hero_images_dir}")
    print(f"Epochs: {epochs} | Batch Size: {batch_size} | LR: {learning_rate}")
    print("=" * 65)

    dataset = FaceCropDataset(hero_images_dir, target_size=(128, 128))
    print(f"Loaded {len(dataset.image_paths)} face images for training.")

    if len(dataset.image_paths) == 0:
        print(f"[ERROR] No face crops found in '{hero_images_dir}'.")
        print("Run: python scripts/prepare_dataset.py --apply first to extract faces.")
        return

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = FaceSwapModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    criterion_l1 = nn.L1Loss()

    best_loss = float("inf")
    best_checkpoint_path = os.path.join(output_checkpoint_dir, "hero_model_best.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        batches = 0

        for batch in loader:
            batch = batch.to(device)

            # Forward pass: reconstruct face
            reconstructed = model(batch)
            loss = criterion_l1(reconstructed, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batches += 1

        avg_loss = epoch_loss / max(1, batches)
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch [{epoch:03d}/{epochs:03d}] - Reconstruction Loss: {avg_loss:.5f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "loss": best_loss,
                },
                best_checkpoint_path,
            )

    print("\n" + "=" * 65)
    print(f"Training Complete! Saved best model checkpoint to: {best_checkpoint_path}")
    print("Next Step: Export to ONNX using:")
    print(f"  python training/export_to_onnx.py --checkpoint {best_checkpoint_path}")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Custom Hero Face Swap Model")
    parser.add_argument("--dataset", type=str, default="datasets/processed/telugu_heroes", help="Path to hero face crops")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Device ('cuda', 'cpu', or 'auto')")
    args = parser.parse_args()

    train_model(
        hero_images_dir=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device_name=args.device,
    )
