"""
Production-Grade Custom Hero Face Swap Model Training Pipeline.
Shared-Encoder + Hero-Decoder Architecture with Perceptual VGG Loss, SSIM Loss,
Data Augmentation, and Live Visual Epoch Previews.

Optimized for NVIDIA GPU training (local or Google Colab / Kaggle).
"""

import os
import sys
import argparse
import time
import glob
import math
from typing import Tuple, Optional, List
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

try:
    import torchvision
    import torchvision.transforms as T
    from torchvision.models import vgg19, VGG19_Weights
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False


# ============================================================================
# 1. MODEL ARCHITECTURE (Shared Encoder + Hero Decoder)
# ============================================================================

class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm2d(channels, affine=True)
        self.leaky = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, x):
        residual = x
        out = self.leaky(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.leaky(out + residual)


class ConvDownBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False)
        self.norm = nn.InstanceNorm2d(out_c, affine=True)
        self.leaky = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.leaky(self.norm(self.conv(x)))


class PixelShuffleUpscale(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c * 4, kernel_size=3, padding=1, bias=False)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.norm = nn.InstanceNorm2d(out_c, affine=True)
        self.leaky = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.leaky(self.norm(self.pixel_shuffle(self.conv(x))))


class SharedEncoder(nn.Module):
    """Encodes face geometry, expression, and pose into a compact identity-agnostic latent code."""
    def __init__(self, in_channels: int = 3, latent_dim: int = 512):
        super().__init__()
        self.down1 = ConvDownBlock(in_channels, 64)    # 128 -> 64
        self.down2 = ConvDownBlock(64, 128)            # 64 -> 32
        self.down3 = ConvDownBlock(128, 256)           # 32 -> 16
        self.down4 = ConvDownBlock(256, 512)           # 16 -> 8
        self.res1 = ResidualBlock(512)
        self.res2 = ResidualBlock(512)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512 * 8 * 8, latent_dim)

    def forward(self, x):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.res1(x)
        x = self.res2(x)
        latent = self.fc(self.flatten(x))
        return latent


class HeroDecoder(nn.Module):
    """Reconstructs the specific hero's photorealistic facial texture and fine details."""
    def __init__(self, latent_dim: int = 512, out_channels: int = 3):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 512 * 8 * 8)
        self.res1 = ResidualBlock(512)
        self.res2 = ResidualBlock(512)
        self.up1 = PixelShuffleUpscale(512, 256)        # 8 -> 16
        self.up2 = PixelShuffleUpscale(256, 128)        # 16 -> 32
        self.up3 = PixelShuffleUpscale(128, 64)         # 32 -> 64
        self.up4 = PixelShuffleUpscale(64, 32)          # 64 -> 128
        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, latent):
        x = F.leaky_relu(self.fc(latent), 0.2)
        x = x.view(-1, 512, 8, 8)
        x = self.res1(x)
        x = self.res2(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        return self.sigmoid(self.out_conv(x))


class FaceSwapModel(nn.Module):
    """Complete End-to-End Face Swap Model (Encoder -> Decoder)."""
    def __init__(self, latent_dim: int = 512):
        super().__init__()
        self.encoder = SharedEncoder(in_channels=3, latent_dim=latent_dim)
        self.decoder = HeroDecoder(latent_dim=latent_dim, out_channels=3)

    def forward(self, source_face):
        latent = self.encoder(source_face)
        swapped_face = self.decoder(latent)
        return swapped_face


# ============================================================================
# 2. PERCEPTUAL & STRUCTURAL LOSS MODULES
# ============================================================================

class SSIMLoss(nn.Module):
    """Structural Similarity Index (SSIM) Loss for high-frequency perceptual fidelity."""
    def __init__(self, window_size: int = 11, channels: int = 3):
        super().__init__()
        self.window_size = window_size
        self.channels = channels
        sigma = 1.5
        gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        gauss = gauss / gauss.sum()
        _1D_window = gauss.unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        self.register_buffer("window", _2D_window.expand(channels, 1, window_size, window_size).contiguous())

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        channel = self.channels
        window = self.window.to(img1.device)
        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=channel)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=channel) - mu1_mu2
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return 1.0 - ssim_map.mean()


class SobelGradientLoss(nn.Module):
    """Measures high-frequency edge difference (eyes, nose, mouth contours) with 0 download overhead."""
    def __init__(self, channels: int = 3):
        super().__init__()
        kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)
        ky = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)
        self.channels = channels

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        gx_p = F.conv2d(pred, self.kx.to(pred.device), padding=1, groups=self.channels)
        gy_p = F.conv2d(pred, self.ky.to(pred.device), padding=1, groups=self.channels)
        gx_t = F.conv2d(target, self.kx.to(target.device), padding=1, groups=self.channels)
        gy_t = F.conv2d(target, self.ky.to(target.device), padding=1, groups=self.channels)
        return F.l1_loss(gx_p, gx_t) + F.l1_loss(gy_p, gy_t)


class VGGPerceptualLoss(nn.Module):
    """VGG19 Feature Loss comparing deep high-level visual activations."""
    def __init__(self, device: torch.device, enabled: bool = False):
        super().__init__()
        self.enabled = False
        if not enabled:
            return
        try:
            weights = VGG19_Weights.DEFAULT
            vgg = vgg19(weights=weights).features.to(device).eval()
            for param in vgg.parameters():
                param.requires_grad = False
            self.slice1 = nn.Sequential(*[vgg[x] for x in range(4)]).to(device)
            self.slice2 = nn.Sequential(*[vgg[x] for x in range(4, 9)]).to(device)
            self.slice3 = nn.Sequential(*[vgg[x] for x in range(9, 14)]).to(device)
            self.slice4 = nn.Sequential(*[vgg[x] for x in range(14, 23)]).to(device)
            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
            self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))
            self.enabled = True
        except Exception as e:
            print(f"[INFO] VGG Perceptual weights unavailable ({e}). Using SSIM + Edge Gradient Loss.")
            self.enabled = False

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return torch.tensor(0.0, device=pred.device)
        p = (pred - self.mean) / self.std
        t = (target - self.mean) / self.std

        h1_p = self.slice1(p)
        h1_t = self.slice1(t)
        h2_p = self.slice2(h1_p)
        h2_t = self.slice2(h1_t)
        h3_p = self.slice3(h2_p)
        h3_t = self.slice3(h2_t)
        h4_p = self.slice4(h3_p)
        h4_t = self.slice4(h3_t)

        loss = (
            0.1 * F.l1_loss(h1_p, h1_t)
            + 0.2 * F.l1_loss(h2_p, h2_t)
            + 0.5 * F.l1_loss(h3_p, h3_t)
            + 1.0 * F.l1_loss(h4_p, h4_t)
        )
        return loss


# ============================================================================
# 3. AUGMENTED DATASET LOADER
# ============================================================================

class AugmentedFaceCropDataset(Dataset):
    """Loads aligned 128x128 face crops with live data augmentations."""
    def __init__(self, image_dir: str, target_size: Tuple[int, int] = (128, 128), augment: bool = True):
        self.image_paths: List[str] = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
            self.image_paths.extend(glob.glob(os.path.join(image_dir, "**", ext), recursive=True))
        self.target_size = target_size
        self.augment = augment

    def __len__(self):
        return max(1, len(self.image_paths))

    def __getitem__(self, idx: int) -> torch.Tensor:
        if not self.image_paths:
            return torch.zeros((3, self.target_size[1], self.target_size[0]), dtype=torch.float32)

        path = self.image_paths[idx % len(self.image_paths)]
        img = cv2.imread(path)
        if img is None:
            return torch.zeros((3, self.target_size[1], self.target_size[0]), dtype=torch.float32)

        img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_LANCZOS4)

        # Live augmentations
        if self.augment:
            # 1. Random horizontal flip
            if np.random.rand() > 0.5:
                img = cv2.flip(img, 1)

            # 2. Slight random rotation (-5 to +5 degrees)
            if np.random.rand() > 0.4:
                angle = np.random.uniform(-5.0, 5.0)
                center = (self.target_size[0] / 2.0, self.target_size[1] / 2.0)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                img = cv2.warpAffine(img, M, self.target_size, borderMode=cv2.BORDER_REFLECT_101)

            # 3. Color & brightness jitter
            if np.random.rand() > 0.4:
                alpha = np.random.uniform(0.90, 1.10)  # Contrast
                beta = np.random.uniform(-10, 10)     # Brightness
                img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        # BGR -> RGB [0.0, 1.0] Tensor NCHW
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img_rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
        return tensor


# ============================================================================
# 4. TRAINING & VALIDATION PIPELINE
# ============================================================================

def save_preview_grid(
    epoch: int,
    model: nn.Module,
    sample_batch: torch.Tensor,
    output_dir: str,
    device: torch.device,
) -> str:
    """Renders a side-by-side [Original | Reconstructed] preview image."""
    model.eval()
    with torch.no_grad():
        reconstructed = model(sample_batch.to(device))

    # Take first 4 faces
    n = min(4, sample_batch.size(0))
    orig_np = sample_batch[:n].permute(0, 2, 3, 1).cpu().numpy()
    rec_np = reconstructed[:n].permute(0, 2, 3, 1).cpu().numpy()

    row_orig = np.hstack([cv2.cvtColor((f * 255).astype(np.uint8), cv2.COLOR_RGB2BGR) for f in orig_np])
    row_rec = np.hstack([cv2.cvtColor((f * 255).astype(np.uint8), cv2.COLOR_RGB2BGR) for f in rec_np])

    grid = np.vstack([row_orig, row_rec])
    # Add title label
    cv2.putText(grid, f"Epoch {epoch:03d} (Top: Target | Bottom: Model)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    out_path = os.path.join(output_dir, f"preview_epoch_{epoch:03d}.jpg")
    cv2.imwrite(out_path, grid)
    return out_path


def train_model(
    hero_images_dir: str,
    output_checkpoint_dir: str = "checkpoints",
    epochs: int = 100,
    batch_size: int = 16,
    learning_rate: float = 2e-4,
    device_name: str = "auto",
    preview_interval: int = 5,
    use_vgg: bool = False,
):
    if not TORCH_AVAILABLE:
        print("[ERROR] PyTorch is required. Install with: pip install torch torchvision")
        return

    os.makedirs(output_checkpoint_dir, exist_ok=True)

    # Device Selection
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    print("=" * 68)
    print("PHOTOREALISTIC HERO FACE SWAP MODEL TRAINING PIPELINE")
    print("=" * 68)
    print(f"Device:           {device}")
    print(f"Dataset:          {hero_images_dir}")
    print(f"Epochs:           {epochs}")
    print(f"Batch Size:       {batch_size}")
    print(f"Learning Rate:    {learning_rate}")
    print(f"Perceptual VGG:   {'ENABLED' if use_vgg else 'DISABLED (Using SSIM + Sobel)'}")
    print(f"Checkpoints Dir:  {output_checkpoint_dir}")
    print("=" * 68)

    dataset = AugmentedFaceCropDataset(hero_images_dir, target_size=(128, 128), augment=True)
    total_imgs = len(dataset.image_paths)
    print(f"Found {total_imgs} face crops in dataset.")
    if total_imgs == 0:
        print(f"[ERROR] No face crops found in '{hero_images_dir}'. Please add face images.")
        return

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True if total_imgs > batch_size else False)

    # Fixed sample batch for visual progression tracking
    sample_dataset = AugmentedFaceCropDataset(hero_images_dir, target_size=(128, 128), augment=False)
    sample_loader = DataLoader(sample_dataset, batch_size=min(4, total_imgs), shuffle=False)
    sample_batch = next(iter(sample_loader))

    model = FaceSwapModel(latent_dim=512).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.5, 0.999), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    criterion_l1 = nn.L1Loss()
    criterion_ssim = SSIMLoss(window_size=11, channels=3).to(device)
    criterion_sobel = SobelGradientLoss(channels=3).to(device)
    criterion_vgg = VGGPerceptualLoss(device, enabled=use_vgg)

    best_loss = float("inf")
    best_checkpoint_path = os.path.join(output_checkpoint_dir, "hero_model_best.pt")

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_l1 = 0.0
        epoch_ssim = 0.0
        epoch_sobel = 0.0
        epoch_vgg = 0.0
        epoch_total = 0.0
        batches = 0

        for batch in loader:
            batch = batch.to(device)

            reconstructed = model(batch)

            loss_l1 = criterion_l1(reconstructed, batch)
            loss_ssim = criterion_ssim(reconstructed, batch)
            loss_sobel = criterion_sobel(reconstructed, batch)
            loss_vgg = criterion_vgg(reconstructed, batch) if use_vgg else torch.tensor(0.0)

            total_loss = loss_l1 + (0.5 * loss_ssim) + (0.3 * loss_sobel) + (0.2 * loss_vgg)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_l1 += loss_l1.item()
            epoch_ssim += loss_ssim.item()
            epoch_sobel += loss_sobel.item()
            epoch_vgg += loss_vgg.item() if isinstance(loss_vgg, torch.Tensor) else 0.0
            epoch_total += total_loss.item()
            batches += 1

        scheduler.step()

        n_b = max(1, batches)
        avg_total = epoch_total / n_b
        avg_l1 = epoch_l1 / n_b
        avg_ssim = epoch_ssim / n_b
        avg_sobel = epoch_sobel / n_b
        avg_vgg = epoch_vgg / n_b

        if epoch % preview_interval == 0 or epoch == 1 or epoch == epochs:
            elapsed = time.time() - start_time
            preview_img = save_preview_grid(epoch, model, sample_batch, output_checkpoint_dir, device)
            print(
                f"Epoch [{epoch:03d}/{epochs:03d}] "
                f"| Loss: {avg_total:.4f} (L1: {avg_l1:.4f}, SSIM: {avg_ssim:.4f}, Edge: {avg_sobel:.4f}) "
                f"| Elapsed: {elapsed:.1f}s "
                f"| Preview: {os.path.basename(preview_img)}"
            )

        if avg_total < best_loss:
            best_loss = avg_total
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_loss,
                },
                best_checkpoint_path,
            )

    print("\n" + "=" * 68)
    print(f"Training Complete! Saved best checkpoint to: {best_checkpoint_path}")
    print("Next Step: Export your trained model to ONNX for 30+ FPS real-time camera use:")
    print(f"  python training/export_to_onnx.py --checkpoint {best_checkpoint_path}")
    print("=" * 68)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Production Hero Face Swap Model")
    parser.add_argument("--dataset", type=str, default="datasets/processed/telugu_heroes", help="Path to hero face crops")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Device ('cuda', 'cpu', or 'auto')")
    parser.add_argument("--preview_interval", type=int, default=5, help="Epoch interval to save preview images")
    parser.add_argument("--use_vgg", action="store_true", help="Enable VGG19 perceptual feature loss (recommended on GPU)")
    args = parser.parse_args()

    train_model(
        hero_images_dir=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device_name=args.device,
        preview_interval=args.preview_interval,
        use_vgg=args.use_vgg,
    )
