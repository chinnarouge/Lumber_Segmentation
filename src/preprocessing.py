from pathlib import Path
import shutil

import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent.parent

TS_ROOT = BASE_DIR / "data" / "Totalsegmentator_dataset_v300"
VERSE_ROOT = BASE_DIR / "data" / "VerSe" / "dataset-verse19test"
AUDIT_DIR = BASE_DIR / "audit"
OUTPUT_DIR = BASE_DIR / "processed"

TARGET_SPACING = (1.5, 1.5, 1.5)
HU_MIN = -1000
HU_MAX = 1500

VAL_RATIO = 0.1
SEED = 42
MAX_CASES = None  # set to a small int for a quick smoke test
MIN_LABEL_VOXELS = 100

TS_LABELS = {
    1: "vertebrae_L1.nii.gz",
    2: "vertebrae_L2.nii.gz",
    3: "vertebrae_L3.nii.gz",
    4: "vertebrae_L4.nii.gz",
    5: "vertebrae_L5.nii.gz",
}

VERSE_LABELS = {
    20: 1,  # L1
    21: 2,  # L2
    22: 3,  # L3
    23: 4,  # L4
    24: 5,  # L5
}


# ============================================================
# Loading
# ============================================================

def load_image(path):
    """Load a NIfTI file and fix small non-orthogonal headers."""

    try:
        return sitk.ReadImage(str(path))

    except RuntimeError as error:
        if "orthonormal direction cosines" not in str(error).lower():
            raise

        nii = nib.load(str(path))
        data = np.asarray(nii.dataobj)

        # Nibabel uses RAS coordinates; SimpleITK uses LPS.
        affine = np.diag([-1, -1, 1, 1]) @ nii.affine

        matrix = affine[:3, :3]
        spacing = np.linalg.norm(matrix, axis=0)
        direction = matrix / spacing

        # Closest orthonormal matrix, kept right-handed.
        u, _, vt = np.linalg.svd(direction)
        direction = u @ vt
        if np.linalg.det(direction) < 0:
            u[:, -1] *= -1
            direction = u @ vt

        # nibabel gives (X, Y, Z); SimpleITK expects (Z, Y, X).
        image = sitk.GetImageFromArray(np.transpose(data, (2, 1, 0)))
        image.SetSpacing(tuple(spacing))
        image.SetOrigin(tuple(affine[:3, 3]))
        image.SetDirection(tuple(direction.flatten()))
        return image


# ============================================================
# Resampling and normalization
# ============================================================

def resample(image, reference=None, is_label=False):
    """Orient to RAS and resample to the target grid."""

    image = sitk.DICOMOrient(image, "RAS")
    interpolator = sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear
    output_pixel_type = sitk.sitkUInt8 if is_label else sitk.sitkFloat32

    if reference is not None:
        return sitk.Resample(
            image, reference, sitk.Transform(), interpolator, 0, output_pixel_type
        )

    old_spacing = np.array(image.GetSpacing())
    old_size = np.array(image.GetSize())
    new_size = np.round(old_size * old_spacing / TARGET_SPACING).astype(int)

    return sitk.Resample(
        image,
        [int(x) for x in new_size],
        sitk.Transform(),
        interpolator,
        image.GetOrigin(),
        TARGET_SPACING,
        image.GetDirection(),
        0,
        output_pixel_type,
    )


def normalize_ct(image):
    """Clip CT values and scale them to [0, 1]."""
    array = sitk.GetArrayFromImage(image).astype(np.float32)
    array = np.clip(array, HU_MIN, HU_MAX)
    return (array - HU_MIN) / (HU_MAX - HU_MIN)


def make_ras_affine(image):
    """Voxel -> RAS mm affine. SimpleITK stores LPS internally."""
    direction = np.array(image.GetDirection()).reshape(3, 3)
    spacing = np.array(image.GetSpacing())

    affine = np.eye(4)
    affine[:3, :3] = direction @ np.diag(spacing)
    affine[:3, 3] = image.GetOrigin()
    return np.diag([-1, -1, 1, 1]) @ affine


# ============================================================
# Shared per-case pipeline
# ============================================================

def process_case(ct_path, load_label, output_path):
    """Preprocess and save one case."""

    ct = resample(load_image(ct_path))
    label = load_label(ct)

    unique_labels = np.unique(label)
    if not np.isin(unique_labels, np.arange(6)).all():
        raise ValueError(f"Unexpected label values: {unique_labels.tolist()}")

    np.savez_compressed(
        output_path,
        image=normalize_ct(ct),
        label=label.astype(np.uint8),
        affine=make_ras_affine(ct),
    )


# ============================================================
# TotalSegmentator
# ============================================================

def ts_load_label(case_dir, reference):
    """Combine TotalSegmentator L1-L5 masks into classes 1-5."""
    label = np.zeros(tuple(reversed(reference.GetSize())), dtype=np.uint8)

    for class_id, filename in TS_LABELS.items():
        mask = resample(
            load_image(case_dir / "segmentations" / filename),
            reference=reference,
            is_label=True,
        )
        label[sitk.GetArrayFromImage(mask) > 0] = class_id

    return label


def get_ts_splits():
    """Train/val/test from audited cases with all five vertebrae present."""
    audit = pd.read_csv(AUDIT_DIR / "totalsegmentator.csv")
    label_columns = [f"L{i}_voxels" for i in range(1, 6)]

    usable = audit[(audit[label_columns] > MIN_LABEL_VOXELS).all(axis=1)]

    train = usable.loc[usable["split"] == "train", "case"].astype(str).to_numpy()
    test = usable.loc[usable["split"] == "test", "case"].astype(str).tolist()

    np.random.default_rng(SEED).shuffle(train)
    n_val = int(len(train) * VAL_RATIO)
    return train[n_val:].tolist(), train[:n_val].tolist(), test


def save_split_ids(train, val, test):
    """Save the complete reproducible split lists."""
    split_dir = OUTPUT_DIR / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for name, cases in [("train", train), ("val", val), ("test", test)]:
        (split_dir / f"{name}.txt").write_text("\n".join(cases))


def process_ts(train, val, test, failures):
    """Preprocess TotalSegmentator cases."""
    for split, cases in [("train", train), ("val", val), ("test", test)]:
        for case_id in tqdm(cases, desc=f"TS {split}"):
            output_path = OUTPUT_DIR / split / f"{case_id}.npz"
            output_path.parent.mkdir(parents=True, exist_ok=True)

            case_dir = TS_ROOT / case_id
            try:
                process_case(
                    case_dir / "ct.nii.gz",
                    lambda ref, d=case_dir: ts_load_label(d, ref),
                    output_path,
                )
            except Exception as error:
                print(f"  FAILED {case_id}: {error}")
                failures.append(f"{split}/{case_id}: {error}")


# ============================================================
# VerSe
# ============================================================

def verse_load_label(seg_path, reference):
    """Convert VerSe labels 20-24 to L1-L5 classes."""
    seg = resample(load_image(seg_path), reference=reference, is_label=True)
    data = sitk.GetArrayFromImage(seg)

    label = np.zeros(data.shape, dtype=np.uint8)
    for verse_id, class_id in VERSE_LABELS.items():
        label[data == verse_id] = class_id
    return label


def find_verse_seg(ct_path):
    """Find the vertebra segmentation for a VerSe CT."""
    name = ct_path.name.replace("_ct.nii.gz", "_seg-vert_msk.nii.gz")
    matches = list((VERSE_ROOT / "derivatives").rglob(name))
    return matches[0] if matches else None


def get_verse_cases():
    """Find VerSe CT volumes with matching segmentations."""
    cases = []
    for ct_path in sorted((VERSE_ROOT / "rawdata").rglob("*_ct.nii.gz")):
        seg_path = find_verse_seg(ct_path)
        if seg_path is None:
            continue
        case_id = ct_path.name.replace("_ct.nii.gz", "")
        cases.append((case_id, ct_path, seg_path))
    return cases


def process_verse(cases, failures):
    """Preprocess the VerSe 2019 test cases."""
    output_dir = OUTPUT_DIR / "verse_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    for case_id, ct_path, seg_path in tqdm(cases, desc="VerSe test"):
        output_path = output_dir / f"{case_id}.npz"

        try:
            process_case(
                ct_path,
                lambda ref, path=seg_path: verse_load_label(path, ref),
                output_path,
            )
        except Exception as error:
            print(f"  FAILED {case_id}: {error}")
            failures.append(f"verse_test/{case_id}: {error}")


# ============================================================
# Output cleanup
# ============================================================

def clear_processed_data():
    """Wipe all generated output before a run so stale .npz files
    from an earlier pipeline version can never be silently reused.
    Original data/ and audit/ files are untouched."""

    for folder in ["train", "val", "test", "verse_test", "splits"]:
        path = OUTPUT_DIR / folder
        if path.exists():
            shutil.rmtree(path)

    failed_path = OUTPUT_DIR / "failed_cases.txt"
    if failed_path.exists():
        failed_path.unlink()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Sanity check
# ============================================================

def sanity_check_affine(case_path):
    """Check an output file and print an L3 centroid."""

    data = np.load(case_path)
    image, label, affine = data["image"], data["label"], data["affine"]

    print("\nSanity check")
    print("------------")
    print(f"File: {case_path.name}")
    print(f"Image shape: {image.shape}")
    print(f"Label shape: {label.shape}")
    print(f"Image range: {image.min():.4f} - {image.max():.4f}")
    print(f"Label classes: {np.unique(label).tolist()}")
    print("Affine:")
    print(affine)

    coords = np.argwhere(label == 3)  # L3
    if len(coords) == 0:
        print("L3 centroid skipped: no L3 label found.")
        return

    z, y, x = coords.mean(axis=0)
    world = affine @ np.array([x, y, z, 1.0])

    print(f"L3 voxel centroid [x,y,z]: {[x, y, z]}")
    print(f"L3 RAS centroid [mm]: {world[:3]}")


# ============================================================
# Main
# ============================================================

def main():
    # Every run starts clean: no stale .npz files from an earlier
    # version of this script can be silently skipped and reused.
    clear_processed_data()

    failures = []

    train, val, test = get_ts_splits()
    save_split_ids(train, val, test)  # save the full split before MAX_CASES trims it

    if MAX_CASES is not None:
        train, val, test = train[:MAX_CASES], val[:MAX_CASES], test[:MAX_CASES]

    print(f"TS train: {len(train)}  val: {len(val)}  test: {len(test)}")
    process_ts(train, val, test, failures)

    verse_cases = get_verse_cases()
    if MAX_CASES is not None:
        verse_cases = verse_cases[:MAX_CASES]

    print(f"VerSe test: {len(verse_cases)}")
    process_verse(verse_cases, failures)

    if failures:
        (OUTPUT_DIR / "failed_cases.txt").write_text("\n".join(failures))
        print(f"\n{len(failures)} cases failed. See failed_cases.txt")
    else:
        print("\nNo preprocessing failures.")

    train_cases = sorted((OUTPUT_DIR / "train").glob("*.npz"))
    if train_cases:
        sanity_check_affine(train_cases[0])

    print("\nPreprocessing complete.")


if __name__ == "__main__":
    main()