import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

from data_loader import NUM_CLASSES
from model import UNet3D
from evaluation import sliding_window_predict, VERTEBRA_NAMES


def extract_vertebral_body(mask, erosion_radius=2):
    """Isolate the vertebral body from posterior elements.

    Erode by `erosion_radius` voxels (≈3 mm at 1.5 mm spacing) to
    disconnect the thin pedicles, then keep the largest connected
    component — always the body.  If erosion removes everything,
    fall back to the largest component of the original mask.
    """
    struct = ndimage.generate_binary_structure(3, 1)
    eroded = ndimage.binary_erosion(mask, struct, iterations=erosion_radius)

    labels, n = ndimage.label(eroded)
    if n == 0:
        labels, n = ndimage.label(mask)
        if n == 0:
            return mask

    sizes = ndimage.sum(mask, labels, range(1, n + 1))
    body_label = np.argmax(sizes) + 1
    return labels == body_label


def _voxel_to_world(voxels, affine):
    cz, cy, cx = voxels.mean(axis=0)
    world = affine @ np.array([cx, cy, cz, 1.0])
    return [round(float(world[0]), 1), round(float(world[1]), 1), round(float(world[2]), 1)]


def compute_centroids(pred, affine, validate=False):
    centroids = {}
    full_centroids = {}
    for c in range(1, NUM_CLASSES):
        mask = pred == c
        if not mask.any():
            continue

        full_centroids[VERTEBRA_NAMES[c]] = _voxel_to_world(np.argwhere(mask), affine)

        body = extract_vertebral_body(mask)
        centroids[VERTEBRA_NAMES[c]] = _voxel_to_world(np.argwhere(body), affine)

    warnings = []
    if validate:
        for name in centroids:
            bc = np.array(centroids[name])
            fc = np.array(full_centroids[name])
            shift = np.linalg.norm(bc - fc)
            if shift < 0.5:
                warnings.append(f"{name}: body centroid identical to full centroid (shift={shift:.1f}mm) — erosion may not have separated body")

        ordered = sorted(centroids.keys(), key=lambda v: int(v[1]))
        for i in range(1, len(ordered)):
            prev, curr = np.array(centroids[ordered[i-1]]), np.array(centroids[ordered[i]])
            dist = np.linalg.norm(curr - prev)
            if dist > 60 or dist < 15:
                warnings.append(f"{ordered[i-1]}-{ordered[i]}: spacing {dist:.1f}mm (expected ~25-40mm)")

    return centroids, full_centroids, warnings


def process_case(model, npz_path, device, use_gt=False):
    with np.load(npz_path) as data:
        image, label, affine = data["image"], data["label"], data["affine"]

    if use_gt:
        seg = label
    else:
        seg = sliding_window_predict(model, image, device)

    return compute_centroids(seg, affine, validate=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--output_dir", default="results/centroids")
    p.add_argument("--base_channels", type=int, default=32)
    p.add_argument("--use_gt", action="store_true", help="Compute from ground-truth labels instead of predictions")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet3D(in_channels=1, num_classes=NUM_CLASSES, base_channels=args.base_channels).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=False))
    model.eval()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cases = sorted(Path(args.data_dir).glob("*.npz"))
    print(f"Computing centroids for {len(cases)} cases ({'GT' if args.use_gt else 'predicted'})...")

    flagged = []
    for npz_path in cases:
        centroids, full_centroids, warnings = process_case(model, npz_path, device, use_gt=args.use_gt)
        result = {
            "case_id": npz_path.stem,
            "coordinate_system": "RAS",
            "unit": "mm",
            "centroids": centroids,
            "full_vertebra_centroids": full_centroids,
        }

        with open(out / f"{npz_path.stem}.json", "w") as f:
            json.dump(result, f, indent=2)

        verts = ", ".join(f"{k}: [{v[0]}, {v[1]}, {v[2]}]" for k, v in centroids.items())
        print(f"  {npz_path.stem}: {verts}")
        if warnings:
            flagged.append(npz_path.stem)
            for w in warnings:
                print(f"    WARNING: {w}")

    print(f"\nDone. {len(cases)} centroid files saved to {out}/")
    if flagged:
        print(f"\n{len(flagged)} cases flagged for review: {flagged}")
