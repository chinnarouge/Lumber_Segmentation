import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from data_loader import NUM_CLASSES
from model import UNet3D
from evaluation import sliding_window_predict
from centroids import compute_centroids

COLORS = ["none", "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231"]
CMAP = ListedColormap(COLORS)
LABELS = ["bg", "L1", "L2", "L3", "L4", "L5"]
CENTROID_COLORS = {"L1": "#e6194b", "L2": "#3cb44b", "L3": "#ffe119", "L4": "#4363d8", "L5": "#f58231"}


def load_and_predict(model, npz_path, device):
    with np.load(npz_path) as data:
        image = data["image"]
        label = data["label"]
        affine = data["affine"]

    pred = sliding_window_predict(model, image, device)
    return image, label, pred, affine


def _world_to_voxel(world_coord, affine):
    inv = np.linalg.inv(affine)
    voxel = inv @ np.array([world_coord[0], world_coord[1], world_coord[2], 1.0])
    return voxel[2], voxel[1], voxel[0]


def plot_case(image, gt, pred, case_id, output_dir, centroids_body=None, centroids_full=None):
    fg_slices = np.where(gt.max(axis=(1, 2)) > 0)[0]
    if len(fg_slices) == 0:
        fg_slices = np.array([image.shape[0] // 2])

    indices = fg_slices[np.linspace(0, len(fg_slices) - 1, min(5, len(fg_slices)), dtype=int)]

    fig, axes = plt.subplots(3, len(indices), figsize=(4 * len(indices), 12))
    if len(indices) == 1:
        axes = axes.reshape(3, 1)

    for col, z in enumerate(indices):
        axes[0, col].imshow(image[z], cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title(f"CT slice {z}")

        axes[1, col].imshow(image[z], cmap="gray", vmin=0, vmax=1)
        mask = np.ma.masked_where(gt[z] == 0, gt[z])
        axes[1, col].imshow(mask, cmap=CMAP, vmin=0, vmax=5, alpha=0.5)
        axes[1, col].set_title("Ground truth")

        axes[2, col].imshow(image[z], cmap="gray", vmin=0, vmax=1)
        mask = np.ma.masked_where(pred[z] == 0, pred[z])
        axes[2, col].imshow(mask, cmap=CMAP, vmin=0, vmax=5, alpha=0.5)
        axes[2, col].set_title("Prediction")

    if centroids_body:
        _draw_centroids_on_nearest(axes, indices, centroids_body, centroids_full)

    for ax in axes.flat:
        ax.axis("off")

    fig.suptitle(case_id, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / f"{case_id}.png", dpi=150)
    plt.close(fig)
    print(f"Saved {case_id}.png")


def _draw_centroids_on_nearest(axes, display_slices, body_centroids, full_centroids):
    for name, voxel in body_centroids.items():
        vz, vy, vx = voxel
        col = int(np.argmin(np.abs(np.array(display_slices) - vz)))
        color = CENTROID_COLORS.get(name, "white")
        axes[2, col].plot(vx, vy, "x", color=color, markersize=14, markeredgewidth=3)
        axes[2, col].annotate(name, (vx, vy), color=color, fontsize=9, fontweight="bold",
                              xytext=(6, -6), textcoords="offset points",
                              bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.6))
        if full_centroids and name in full_centroids:
            fz, fy, fx = full_centroids[name]
            axes[2, col].plot(fx, fy, "o", color=color, markersize=10, markeredgewidth=2, fillstyle="none")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--data_dir", default="processed/val")
    p.add_argument("--output_dir", default="visualizations")
    p.add_argument("--max_cases", type=int, default=5)
    p.add_argument("--base_channels", type=int, default=32)
    p.add_argument("--centroids", action="store_true", help="Overlay vertebral body centroids on prediction")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet3D(in_channels=1, num_classes=NUM_CLASSES, base_channels=args.base_channels).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=False))
    model.eval()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cases = sorted(Path(args.data_dir).glob("*.npz"))[:args.max_cases]
    for npz in cases:
        image, gt, pred, affine = load_and_predict(model, npz, device)
        body_voxels = None
        full_voxels = None
        if args.centroids:
            body_c, full_c, warnings = compute_centroids(pred, affine, validate=True)
            body_voxels = {n: _world_to_voxel(c, affine) for n, c in body_c.items()}
            full_voxels = {n: _world_to_voxel(c, affine) for n, c in full_c.items()}
            if warnings:
                for w in warnings:
                    print(f"  WARNING ({npz.stem}): {w}")
        plot_case(image, gt, pred, npz.stem, out, body_voxels, full_voxels)

    print(f"\nDone. {len(cases)} images saved to {out}/")
