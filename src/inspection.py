from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent.parent
TS_ROOT = BASE_DIR / "data" / "Totalsegmentator_dataset_v300"
VERSE_ROOT = BASE_DIR / "data" / "VerSe" / "dataset-verse19test"
OUT_DIR = BASE_DIR / "audit"

OUT_DIR.mkdir(exist_ok=True)

TS_LABELS = {
    "L1": "vertebrae_L1.nii.gz",
    "L2": "vertebrae_L2.nii.gz",
    "L3": "vertebrae_L3.nii.gz",
    "L4": "vertebrae_L4.nii.gz",
    "L5": "vertebrae_L5.nii.gz",
}

VERSE_LABELS = {
    20: "L1",
    21: "L2",
    22: "L3",
    23: "L4",
    24: "L5",
    25: "L6",
}

DIRECTION_THRESHOLD = 1e-3
MIN_LABEL_VOXELS = 100


def spacing(affine):
    return np.linalg.norm(affine[:3, :3], axis=0)


def orientation(affine):
    return "".join(nib.aff2axcodes(affine))


def direction_error(affine):
    matrix = affine[:3, :3]
    scales = np.linalg.norm(matrix, axis=0)

    if np.any(scales == 0):
        return np.inf

    direction = matrix / scales
    return float(np.max(np.abs(direction.T @ direction - np.eye(3))))


def valid_affine(affine):
    return (
        np.isfinite(affine).all()
        and abs(np.linalg.det(affine[:3, :3])) > 1e-8
    )


def intensity_stats(data):
    data = data[np.isfinite(data)]

    if data.size == 0:
        return [np.nan] * 6

    return [
        float(np.min(data)),
        float(np.max(data)),
        float(np.percentile(data, 1)),
        float(np.percentile(data, 5)),
        float(np.percentile(data, 50)),
        float(np.percentile(data, 99)),
    ]


def inspect_label(ct, label_path, name, anomalies):
    row = {}

    if not label_path.exists():
        anomalies.append(f"missing_{name}_label")
        row[f"{name}_voxels"] = 0
        row[f"{name}_shape_match"] = False
        row[f"{name}_affine_match"] = False
        row[f"{name}_direction_error"] = np.nan
        return row

    label = nib.load(label_path)
    data = np.asarray(label.dataobj)

    row[f"{name}_voxels"] = int(np.sum(data > 0))
    row[f"{name}_shape_match"] = label.shape == ct.shape
    row[f"{name}_affine_match"] = np.allclose(
        label.affine, ct.affine, atol=1e-4
    )
    row[f"{name}_direction_error"] = direction_error(label.affine)

    if not row[f"{name}_shape_match"]:
        anomalies.append(f"{name}_shape_mismatch")

    if not row[f"{name}_affine_match"]:
        anomalies.append(f"{name}_affine_mismatch")

    if not valid_affine(label.affine):
        anomalies.append(f"invalid_{name}_affine")

    if row[f"{name}_direction_error"] > DIRECTION_THRESHOLD:
        anomalies.append(f"non_orthogonal_{name}_affine")

    if np.isnan(data).any():
        anomalies.append(f"{name}_nan")

    if np.isinf(data).any():
        anomalies.append(f"{name}_inf")

    if not np.all(np.isin(np.unique(data), [0, 1])):
        anomalies.append(f"unexpected_{name}_values")

    if row[f"{name}_voxels"] <= MIN_LABEL_VOXELS:
        anomalies.append(f"{name}_small")

    return row


def find_verse_seg(ct_path):
    name = ct_path.name.replace(
        "_ct.nii.gz",
        "_seg-vert_msk.nii.gz",
    )

    matches = list(
        (VERSE_ROOT / "derivatives").rglob(name)
    )

    return matches[0] if matches else None


def inspect_case(ct_path, label_map=None):
    case = ct_path.name.replace("_ct.nii.gz", "")
    if case == "ct.nii.gz":
        case = ct_path.parent.name
    row = {
        "case": case,
        "error": "",
        "shape": "",
        "orientation": "",
        "spacing": "",
        "direction_error": np.nan,
        "nan": False,
        "inf": False,
        "intensity_min": np.nan,
        "intensity_max": np.nan,
        "intensity_p1": np.nan,
        "intensity_p5": np.nan,
        "intensity_p50": np.nan,
        "intensity_p99": np.nan,
        "unexpected_labels": "",
        "anomalies": "",
    }

    anomalies = []

    try:
        ct = nib.load(ct_path)
        data = np.asarray(ct.dataobj)

        # Raw-file orientation and spacing vary by scanner and are not
        # anomalies on their own; the pipeline reorients/resamples them.
        row["shape"] = str(ct.shape)
        row["orientation"] = orientation(ct.affine)
        row["spacing"] = str(tuple(np.round(spacing(ct.affine), 4)))
        row["direction_error"] = direction_error(ct.affine)

        row["nan"] = bool(np.isnan(data).any())
        row["inf"] = bool(np.isinf(data).any())

        stats = intensity_stats(data)
        (
            row["intensity_min"],
            row["intensity_max"],
            row["intensity_p1"],
            row["intensity_p5"],
            row["intensity_p50"],
            row["intensity_p99"],
        ) = stats

        if not valid_affine(ct.affine):
            anomalies.append("invalid_ct_affine")

        if row["direction_error"] > DIRECTION_THRESHOLD:
            anomalies.append("non_orthogonal_ct_affine")

        if row["nan"]:
            anomalies.append("ct_nan")

        if row["inf"]:
            anomalies.append("ct_inf")

        if label_map == "totalsegmentator":
            for name, filename in TS_LABELS.items():
                row.update(
                    inspect_label(
                        ct,
                        ct_path.parent / "segmentations" / filename,
                        name,
                        anomalies,
                    )
                )

        elif label_map == "verse":
            seg_path = find_verse_seg(ct_path)

            if seg_path is None:
                anomalies.append("missing_segmentation")
            else:
                seg = nib.load(seg_path)
                seg_data = np.asarray(seg.dataobj)

                if seg.shape != ct.shape:
                    anomalies.append("seg_shape_mismatch")

                if not np.allclose(
                    seg.affine,
                    ct.affine,
                    atol=1e-4,
                ):
                    anomalies.append("seg_affine_mismatch")

                seg_direction = direction_error(seg.affine)

                if not valid_affine(seg.affine):
                    anomalies.append("invalid_seg_affine")

                if seg_direction > DIRECTION_THRESHOLD:
                    anomalies.append("non_orthogonal_seg_affine")

                if np.isnan(seg_data).any():
                    anomalies.append("seg_nan")

                if np.isinf(seg_data).any():
                    anomalies.append("seg_inf")

                unique_labels = np.unique(seg_data)

                unexpected = [
                    int(x)
                    for x in unique_labels
                    if x != 0 and x not in range(1,29)
                ]

                if unexpected:
                    row["unexpected_labels"] = ",".join(
                        map(str, unexpected)
                    )
                    anomalies.append("unexpected_labels")

                for label_id, name in VERSE_LABELS.items():
                    voxels = int(np.sum(seg_data == label_id))
                    row[f"{name}_voxels"] = voxels

                row["has_L6"] = row.get("L6_voxels", 0) > 0

    except Exception as e:
        row["error"] = str(e)
        anomalies.append("read_error")

    row["anomalies"] = ";".join(sorted(set(anomalies)))
    return row


def audit_totalsegmentator():
    meta = pd.read_csv(TS_ROOT / "meta.csv", sep=";")

    if meta["image_id"].duplicated().any():
        print("Warning: duplicate image_id values in meta.csv")

    cases = sorted(
        p for p in TS_ROOT.iterdir()
        if p.is_dir() and (p / "ct.nii.gz").exists()
    )

    rows = []

    for case_dir in tqdm(cases, desc="TotalSegmentator"):
        row = inspect_case(
            case_dir / "ct.nii.gz",
            label_map="totalsegmentator",
        )

        match = meta[meta["image_id"].astype(str) == case_dir.name]

        if len(match) == 1:
            row["split"] = match.iloc[0]["split"]
        else:
            row["split"] = ""
            row["anomalies"] = (row["anomalies"] + ";metadata_missing").lstrip(";")

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "totalsegmentator.csv", index=False)

    return df


def audit_verse():
    ct_files = sorted(
        (VERSE_ROOT / "rawdata").rglob("*_ct.nii.gz")
    )

    rows = []

    for ct_path in tqdm(ct_files, desc="VerSe"):
        row = inspect_case(
            ct_path,
            label_map="verse",
        )

        row["split"] = "test"
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "verse.csv", index=False)

    return df


def main():
    ts = audit_totalsegmentator()
    verse = audit_verse()

    print("\nAudit complete")
    print(f"TotalSegmentator: {len(ts)} cases")
    print(f"VerSe: {len(verse)} cases")
    print(f"Output: {OUT_DIR}")


if __name__ == "__main__":
    main()