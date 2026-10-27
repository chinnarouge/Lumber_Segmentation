import argparse
import json
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import torch
import torch.nn.functional as F
import numpy as np
import wandb

from data_loader import get_dataloader, NUM_CLASSES, PATCH_SIZE
from model import UNet3D

VERTEBRA_NAMES = {1: "L1", 2: "L2", 3: "L3", 4: "L4", 5: "L5"}

LABEL_MAPPING = {
    "totalsegmentator": {
        "source_format": "Separate binary NIfTI mask per vertebra (vertebrae_L1.nii.gz, ...)",
        "mapping": {"vertebrae_L1": 1, "vertebrae_L2": 2, "vertebrae_L3": 3, "vertebrae_L4": 4, "vertebrae_L5": 5},
        "notes": "Binary masks merged into single label volume during preprocessing.",
    },
    "verse": {
        "source_format": "Single segmentation NIfTI with integer labels per vertebra",
        "mapping": {"20 (L1)": 1, "21 (L2)": 2, "22 (L3)": 3, "23 (L4)": 4, "24 (L5)": 5},
        "ignored_labels": {"25": "L6 (transitional vertebra) — mapped to background (0)"},
        "notes": "VerSe label 25 (L6) present in some cases, excluded from evaluation.",
    },
}


def sliding_window_predict(model, image, device, patch_size=PATCH_SIZE, overlap=0.5):
    d, h, w = image.shape
    pd, ph, pw = patch_size
    sd, sh, sw = int(pd * (1 - overlap)), int(ph * (1 - overlap)), int(pw * (1 - overlap))

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


def dice_per_class(logits, target, num_classes=6, eps=1e-5):
    preds = logits.argmax(dim=1)
    scores = {}
    for c in range(1, num_classes):
        p = (preds == c).float()
        t = (target == c).float()
        inter = (p * t).sum()
        union = p.sum() + t.sum()
        scores[c] = ((2 * inter + eps) / (union + eps)).item()
    return scores


def evaluate_loader(model, loader, device, num_classes=6):
    model.eval()
    all_scores = {c: [] for c in range(1, num_classes)}
    case_results = []
    total_loss = 0.0

    with torch.no_grad():
        for image, label, case_ids in loader:
            image, label = image.to(device), label.to(device)
            logits = model(image)

            ce = F.cross_entropy(logits, label)
            probs = F.softmax(logits, dim=1)
            target_oh = F.one_hot(label, num_classes).permute(0, 4, 1, 2, 3).float()
            dims = (0, 2, 3, 4)
            inter = (probs * target_oh).sum(dims)
            union = probs.sum(dims) + target_oh.sum(dims)
            dl = 1 - ((2 * inter + 1e-5) / (union + 1e-5))[1:].mean()
            total_loss += (ce + dl).item()

            scores = dice_per_class(logits, label, num_classes)
            for c, s in scores.items():
                all_scores[c].append(s)
            case_results.append({"case_id": case_ids[0], **{VERTEBRA_NAMES[c]: s for c, s in scores.items()}})

    mean_dice = {VERTEBRA_NAMES[c]: np.mean(v) for c, v in all_scores.items()}
    mean_dice["mean"] = np.mean(list(mean_dice.values()))
    avg_loss = total_loss / len(loader)
    return mean_dice, case_results, avg_loss


def evaluate_full_volume(model, data_dir, device, num_classes=6):
    model.eval()
    paths = sorted(Path(data_dir).glob("*.npz"))
    all_scores = {c: [] for c in range(1, num_classes)}
    case_results = []

    for npz_path in paths:
        with np.load(npz_path) as data:
            image, label = data["image"], data["label"]

        pred = sliding_window_predict(model, image, device)
        scores = {}
        case_info = {"case_id": npz_path.stem}
        for c in range(1, num_classes):
            p = (pred == c).astype(np.float32)
            t = (label == c).astype(np.float32)
            gt_voxels = t.sum()
            pred_voxels = p.sum()

            if gt_voxels == 0 and pred_voxels == 0:
                case_info[VERTEBRA_NAMES[c]] = "absent"
                continue
            if gt_voxels == 0:
                case_info[VERTEBRA_NAMES[c]] = "no_reference"
                continue

            inter = (p * t).sum()
            union = p.sum() + t.sum()
            scores[c] = float((2 * inter + 1e-5) / (union + 1e-5))
            all_scores[c].append(scores[c])
            case_info[VERTEBRA_NAMES[c]] = scores[c]

        case_results.append(case_info)
        valid_scores = list(scores.values())
        mean_s = np.mean(valid_scores) if valid_scores else 0.0
        print(f"  {npz_path.stem}: mean={mean_s:.4f} ({len(valid_scores)}/5 vertebrae scored)")

    scored_counts = {}
    absent_counts = {}
    false_positive_counts = {}
    for c in range(1, num_classes):
        name = VERTEBRA_NAMES[c]
        scored_counts[name] = len(all_scores[c])
        absent_counts[name] = sum(1 for r in case_results if r.get(name) == "absent")
        false_positive_counts[name] = sum(1 for r in case_results if r.get(name) == "no_reference")

    mean_dice = {}
    classes_with_scores = []
    for c in range(1, num_classes):
        name = VERTEBRA_NAMES[c]
        if all_scores[c]:
            mean_dice[name] = float(np.mean(all_scores[c]))
            classes_with_scores.append(mean_dice[name])
        else:
            mean_dice[name] = None
    mean_dice["mean"] = float(np.mean(classes_with_scores)) if classes_with_scores else None

    summary = {
        "total_cases": len(case_results),
        "per_vertebra": {
            name: {
                "eligible": scored_counts[name],
                "absent": absent_counts[name],
                "no_reference_predicted": false_positive_counts[name],
            }
            for name in ["L1", "L2", "L3", "L4", "L5"]
        },
    }
    return mean_dice, case_results, summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True, help="Path to best_model.pt")
    p.add_argument("--ts_test_dir", default="processed/test")
    p.add_argument("--verse_test_dir", default="processed/verse_test")
    p.add_argument("--base_channels", type=int, default=32)
    p.add_argument("--output_dir", default="results")
    p.add_argument("--run_name", default=None)
    p.add_argument("--wandb_project", default="lumbar-segmentation")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet3D(in_channels=1, num_classes=NUM_CLASSES, base_channels=args.base_channels).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=False))

    run_name = args.run_name or f"eval_{Path(args.model_path).parent.name}"
    wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_results = {"label_mapping": LABEL_MAPPING}

    for name, data_dir in [("totalsegmentator_test", args.ts_test_dir), ("verse_test", args.verse_test_dir)]:
        test_dir = Path(data_dir)
        if not test_dir.exists() or not list(test_dir.glob("*.npz")):
            print(f"Skipping {name}: no data in {data_dir}")
            continue

        print(f"\nEvaluating {name}...")
        mean_dice, case_results, summary = evaluate_full_volume(model, data_dir, device, NUM_CLASSES)

        print(f"\n{'='*40}")
        print(f"{name} ({summary['total_cases']} cases)")
        print(f"{'='*40}")
        for v in ["L1", "L2", "L3", "L4", "L5"]:
            info = summary["per_vertebra"][v]
            val = f"{mean_dice[v]:.4f}" if mean_dice[v] is not None else "N/A"
            print(f"  {v}: {val}  (eligible={info['eligible']}, absent={info['absent']}, no_ref_predicted={info['no_reference_predicted']})")
        val = f"{mean_dice['mean']:.4f}" if mean_dice['mean'] is not None else "N/A"
        print(f"  Mean: {val}")

        dataset_result = {"mean_dice": mean_dice, "per_case": case_results, "summary": summary}
        log_dice = {f"{name}/{v}": mean_dice[v] for v in ["L1", "L2", "L3", "L4", "L5", "mean"] if mean_dice[v] is not None}
        wandb.log(log_dice)
        all_results[name] = dataset_result

        with open(out / f"{name}_dice.json", "w") as f:
            json.dump(dataset_result, f, indent=2)

    with open(out / "cross_dataset_evaluation.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out}/")

    wandb.finish()
