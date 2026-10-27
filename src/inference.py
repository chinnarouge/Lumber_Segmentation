import argparse
import json
from pathlib import Path

import numpy as np
import nibabel as nib
import SimpleITK as sitk
import torch
from scipy import ndimage

from model import UNet3D

TARGET_SPACING = (1.5, 1.5, 1.5)
HU_MIN, HU_MAX = -1000, 1500
PATCH_SIZE = (128, 128, 128)
NUM_CLASSES = 6
BASE_CHANNELS = 32
VERTEBRA_NAMES = {1: "L1", 2: "L2", 3: "L3", 4: "L4", 5: "L5"}


def load_image(path):
    try:
        return sitk.ReadImage(str(path))
    except RuntimeError as error:
        if "orthonormal direction cosines" not in str(error).lower():
            raise
        nii = nib.load(str(path))
        data = np.asarray(nii.dataobj)
        affine = np.diag([-1, -1, 1, 1]) @ nii.affine
        matrix = affine[:3, :3]
        spacing = np.linalg.norm(matrix, axis=0)
        direction = matrix / spacing
        u, _, vt = np.linalg.svd(direction)
        direction = u @ vt
        if np.linalg.det(direction) < 0:
            u[:, -1] *= -1
            direction = u @ vt
        image = sitk.GetImageFromArray(np.transpose(data, (2, 1, 0)))
        image.SetSpacing(tuple(spacing))
        image.SetOrigin(tuple(affine[:3, 3]))
        image.SetDirection(tuple(direction.flatten()))
        return image


def resample(image):
    image = sitk.DICOMOrient(image, "RAS")
    old_spacing = np.array(image.GetSpacing())
    old_size = np.array(image.GetSize())
    new_size = np.round(old_size * old_spacing / TARGET_SPACING).astype(int)
    return sitk.Resample(
        image, [int(x) for x in new_size], sitk.Transform(),
        sitk.sitkLinear, image.GetOrigin(), TARGET_SPACING,
        image.GetDirection(), 0, sitk.sitkFloat32,
    )


def normalize_ct(image):
    array = sitk.GetArrayFromImage(image).astype(np.float32)
    array = np.clip(array, HU_MIN, HU_MAX)
    return (array - HU_MIN) / (HU_MAX - HU_MIN)


def make_ras_affine(image):
    direction = np.array(image.GetDirection()).reshape(3, 3)
    spacing = np.array(image.GetSpacing())
    affine = np.eye(4)
    affine[:3, :3] = direction @ np.diag(spacing)
    affine[:3, 3] = image.GetOrigin()
    return np.diag([-1, -1, 1, 1]) @ affine


def sliding_window_predict(model, image, device):
    d, h, w = image.shape
    pd, ph, pw = PATCH_SIZE
    sd, sh, sw = pd // 2, ph // 2, pw // 2

    def starts(dim, p, s):
        st = list(range(0, max(dim - p + 1, 1), s))
        if st[-1] + p < dim:
            st.append(max(dim - p, 0))
        return st

    pred_sum = np.zeros((NUM_CLASSES, d, h, w), dtype=np.float32)
    count = np.zeros((d, h, w), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for dz in starts(d, pd, sd):
            for dy in starts(h, ph, sh):
                for dx in starts(w, pw, sw):
                    patch = image[dz:dz+pd, dy:dy+ph, dx:dx+pw]
                    ad, ah, aw = patch.shape
                    if ad < pd or ah < ph or aw < pw:
                        patch = np.pad(patch, ((0, pd-ad), (0, ph-ah), (0, pw-aw)))
                    inp = torch.from_numpy(patch.copy()).unsqueeze(0).unsqueeze(0).float().to(device)
                    probs = torch.softmax(model(inp), dim=1).squeeze(0).cpu().numpy()
                    pred_sum[:, dz:dz+ad, dy:dy+ah, dx:dx+aw] += probs[:, :ad, :ah, :aw]
                    count[dz:dz+ad, dy:dy+ah, dx:dx+aw] += 1

    pred_sum /= np.maximum(count, 1)
    return pred_sum.argmax(axis=0).astype(np.uint8)


def extract_vertebral_body(mask, erosion_radius=2):
    struct = ndimage.generate_binary_structure(3, 1)
    eroded = ndimage.binary_erosion(mask, struct, iterations=erosion_radius)
    labels, n = ndimage.label(eroded)
    if n == 0:
        labels, n = ndimage.label(mask)
        if n == 0:
            return mask
    sizes = ndimage.sum(mask, labels, range(1, n + 1))
    return labels == (np.argmax(sizes) + 1)


def compute_centroids(pred, affine):
    centroids = {}
    for c in range(1, NUM_CLASSES):
        mask = pred == c
        if not mask.any():
            continue
        body = extract_vertebral_body(mask)
        cz, cy, cx = np.argwhere(body).mean(axis=0)
        world = affine @ np.array([cx, cy, cz, 1.0])
        centroids[VERTEBRA_NAMES[c]] = [round(float(world[0]), 1),
                                         round(float(world[1]), 1),
                                         round(float(world[2]), 1)]
    return centroids


def main():
    p = argparse.ArgumentParser(description="Lumbar spine segmentation inference")
    p.add_argument("input_dir", help="Directory containing NIfTI CT scans")
    p.add_argument("output_dir", help="Directory for output segmentations and centroids")
    p.add_argument("--model_path", default=None, help="Path to model weights (auto-detected if omitted)")
    args = p.parse_args()

    if args.model_path is None:
        args.model_path = Path(__file__).resolve().parent.parent / "checkpoints" / "full_v2" / "best_model.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet3D(in_channels=1, num_classes=NUM_CLASSES, base_channels=BASE_CHANNELS).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=False))
    model.eval()
    print(f"Model loaded from {args.model_path} (device: {device})")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    niftis = sorted(input_dir.rglob("*.nii.gz")) + sorted(input_dir.rglob("*.nii"))
    niftis = [f for f in niftis if "seg" not in f.name and "msk" not in f.name]
    if not niftis:
        print(f"No NIfTI files found in {input_dir}")
        return

    print(f"Processing {len(niftis)} cases...")

    for nii_path in niftis:
        case_id = nii_path.name.replace(".nii.gz", "").replace(".nii", "")
        print(f"  {case_id}...", end=" ", flush=True)

        ct = resample(load_image(nii_path))
        image = normalize_ct(ct)
        affine = make_ras_affine(ct)

        pred = sliding_window_predict(model, image, device)

        seg_sitk = sitk.GetImageFromArray(pred.astype(np.uint8))
        seg_sitk.CopyInformation(ct)
        sitk.WriteImage(seg_sitk, str(output_dir / f"{case_id}_seg.nii.gz"))

        centroids = compute_centroids(pred, affine)
        result = {
            "case_id": case_id,
            "coordinate_system": "RAS",
            "unit": "mm",
            "centroids": centroids,
        }
        with open(output_dir / f"{case_id}_centroids.json", "w") as f:
            json.dump(result, f, indent=2)

        print(f"{len(centroids)} vertebrae")

    print(f"\nDone. Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
