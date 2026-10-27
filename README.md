# Lumbar Spine Segmentation from CT

This project segments the five lumbar vertebrae (L1-L5) from 3D CT scans and
computes vertebral body centroids in RAS world coordinates. It contains the
complete preprocessing, training, evaluation, centroid extraction, and
visualization pipeline described in the accompanying assignment report.

The model is a four-level 3D U-Net trained on complete lumbar annotations from
TotalSegmentator v3. It is evaluated on a held-out TotalSegmentator test split
and on VerSe 2019 as an out-of-domain dataset.

> **Scope and safety:** This is a research/assignment implementation, not a
> clinically validated medical device. Predictions require review by a
> qualified medical professional before any clinical use.

## Results

The final evaluation uses full-volume sliding-window inference with a
`128 x 128 x 128` patch and 50% overlap. Dice is the unweighted mean of the
per-case scores for each vertebra; a vertebra with no reference label is
excluded, while a missed vertebra with an eligible reference contributes zero.

| Vertebra | TotalSegmentator test (n=57) | VerSe 2019 (eligible cases) |
|---|---:|---:|
| L1 | 0.798 | 0.815 (n=36) |
| L2 | 0.834 | 0.863 (n=33) |
| L3 | 0.868 | 0.869 (n=32) |
| L4 | 0.864 | 0.751 (n=31) |
| L5 | 0.746 | 0.696 (n=31) |
| **Mean** | **0.822** | **0.799** |

The TotalSegmentator score covers 57 of its 109 official test cases because
only cases with all five lumbar labels (more than 100 voxels per label) are
used. VerSe has variable eligibility by vertebra because some cases do not
contain every lumbar reference label.

## Quick start: run inference

### Input and output

The inference script recursively finds `.nii.gz` and `.nii` CT files in the
input directory. Files whose names contain `seg` or `msk` are ignored.

For an input named `patient01.nii.gz`, the output directory contains:

- `patient01_seg.nii.gz`: an unsigned 8-bit segmentation mask
  (`0` background, `1` L1, ..., `5` L5).
- `patient01_centroids.json`: body-centroid coordinates in RAS millimetres.

Example centroid file:

```json
{
  "case_id": "patient01",
  "coordinate_system": "RAS",
  "unit": "mm",
  "centroids": {
    "L1": [2.3, -45.6, 78.9],
    "L2": [2.1, -44.8, 49.2]
  }
}
```

### Option A: Docker (recommended)

Docker packages the inference dependencies and the trained checkpoint:

```bash
docker build -t lumbar-seg .
./run_inference.sh /path/to/ct_scans /path/to/output
```

The Docker path requires Docker, an NVIDIA GPU, and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).
The container uses the bundled
`checkpoints/full_v2/best_model.pt` checkpoint.

### Option B: local Python

Use Python 3.10 or newer. A CUDA GPU is recommended; CPU inference is
supported but may be slow.

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/inference.py /path/to/ct_scans /path/to/output
```

The shell wrapper automatically uses the `lumbar-seg` Docker image when that
image exists and otherwise falls back to local Python:

```bash
./run_inference.sh /path/to/ct_scans /path/to/output
```

To use another checkpoint:

```bash
python src/inference.py /path/to/ct_scans /path/to/output \
  --model_path /path/to/best_model.pt
```

## Reproduce preprocessing and training

### 1. Obtain the datasets

The datasets are not included in this repository. Arrange them as follows:

```text
code/
├── data/
│   ├── Totalsegmentator_dataset_v300/
│   └── VerSe/
│       └── dataset-verse19test/
```

The preprocessing script expects:

- TotalSegmentator CT volumes at
  `data/Totalsegmentator_dataset_v300/<case>/ct.nii.gz`.
- TotalSegmentator binary masks at
  `.../<case>/segmentations/vertebrae_L1.nii.gz` through
  `vertebrae_L5.nii.gz`.
- VerSe raw CT files under `data/VerSe/dataset-verse19test/rawdata/`.
- Matching VerSe segmentations under the corresponding `derivatives/` tree.

### 2. Audit and preprocess

The included audit CSVs record the dataset checks used by the assignment.
To regenerate preprocessing outputs from the raw datasets:

```bash
python src/preprocessing.py
```

The pipeline:

1. Reorients images to RAS.
2. Resamples CT and labels to 1.5 mm isotropic spacing.
3. Clips CT values to [-1000, 1500] HU and normalizes to [0, 1].
4. Harmonizes labels to `0` background and `1-5` for L1-L5.
5. Saves compressed NPZ files containing `image`, `label`, and `affine`.

TotalSegmentator cases are filtered to those with all five labels present.
The reproducible split is saved in `processed/splits/` as 667 training, 74
validation, and 57 test cases. VerSe cases are processed into
`processed/verse_test/` for external evaluation. Each run clears generated
processed outputs first; the original `data/` and `audit/` files are left
untouched.

The loader uses foreground-centered random `128^3` crops during training and
center crops for the inexpensive per-epoch validation signal.

### 3. Train

Training uses a 3D U-Net with approximately 18.8 million parameters,
cross-entropy plus foreground Dice loss, Adam, a ReduceLROnPlateau scheduler,
and early stopping. Weights & Biases is used for experiment logging.

Create a `.env` file in the project root (never commit it):

```text
WANDB_API_KEY=your_key_here
```

Example command matching the reported run:

```bash
python src/train.py \
  --epochs 50 \
  --batch_size 4 \
  --lr 0.01 \
  --base_channels 32 \
  --run_name full_v2
```

The reported run used an NVIDIA L4 (16 GB), took about seven minutes per
epoch, and selected its best checkpoint at epoch 48. Checkpoints are written
under `checkpoints/<run_name>/`.

> **Important evaluation note:** training-time validation uses a center crop,
> while final evaluation uses full-volume sliding windows. These are different
> measurements; the reported center-crop validation Dice (~0.66) should not be
> compared directly with the final full-volume Dice (~0.82).

### 4. Evaluate

```bash
python src/evaluation.py \
  --model_path checkpoints/full_v2/best_model.pt \
  --base_channels 32 \
  --output_dir results
```

The script evaluates both `processed/test` and `processed/verse_test` when
they exist, writes per-case and summary JSON files to `results/`, and logs
metrics to Weights & Biases.

### 5. Extract vertebral body centroids

Both source datasets label the full vertebra rather than the vertebral body.
The centroid implementation approximates the body by eroding each predicted
vertebra by two voxels (about 3 mm), selecting the largest connected
component, and converting its centroid to RAS millimetres. It also stores the
full-vertebra centroid and warns about suspicious inter-vertebra spacing.

Predicted centroids:

```bash
python src/centroids.py \
  --model_path checkpoints/full_v2/best_model.pt \
  --data_dir processed/test \
  --output_dir results/centroids
```

Ground-truth centroids for comparison:

```bash
python src/centroids.py \
  --model_path checkpoints/full_v2/best_model.pt \
  --data_dir processed/test \
  --output_dir results/centroids_gt \
  --use_gt
```

The erosion method is a morphological approximation and has not been
validated against manual body annotations. Review flagged cases before using
their centroids.

### 6. Visualize predictions

```bash
python src/visualize.py \
  --model_path checkpoints/full_v2/best_model.pt \
  --data_dir processed/test \
  --output_dir visualizations \
  --max_cases 5 \
  --centroids
```

Each PNG shows CT slices, ground truth overlays, and prediction overlays.
With `--centroids`, `x` marks the eroded body centroid and an open circle
marks the full-vertebra centroid.

## Project layout

```text
src/
├── centroids.py       Body and full-vertebra centroid extraction
├── data_loader.py     NPZ dataset and foreground-centered crops
├── evaluation.py      Sliding-window inference and Dice evaluation
├── inference.py       Standalone NIfTI inference entry point
├── inspection.py      Raw dataset audit and anomaly detection
├── model.py           Four-level 3D U-Net
├── preprocessing.py   Reorientation, resampling, normalization, label mapping
├── train.py           Training loop and checkpointing
└── visualize.py       Segmentation and centroid overlays
├── Dockerfile         Reproducible inference image
├── requirements.txt   Python dependencies
└── run_inference.sh   Docker/local inference wrapper
```

The minimum files needed for inference are `src/inference.py`,
`src/model.py`, and `checkpoints/full_v2/best_model.pt`. Training and
evaluation additionally require the processed datasets and the full Python
dependencies.

## Known limitations and future work

- The model is trained on about 700 usable cases and shows a train-validation
  gap, so augmentation (rotations, elastic deformations, and intensity
  shifts) is a priority.
- The L4/L5 VerSe reduction may reflect lumbosacral anatomical variation,
  transitional vertebrae, or numbering ambiguity; it needs case-by-case
  review.
- The centroid heuristic needs validation against manual annotations with
  millimetre-level error reporting.
- Filtering to complete L1-L5 annotations excludes 52 official
  TotalSegmentator test cases. A future evaluation should handle
  per-vertebra missing labels explicitly.

## License and data

No separate license has been assigned to this assignment code. Add an
appropriate license before redistributing it. Dataset terms and access
conditions are governed by the respective TotalSegmentator and VerSe
licenses; do not commit raw patient data or restricted datasets.
