from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

NUM_CLASSES = 6  # background + L1-L5
PATCH_SIZE = (128, 128, 128)


class LumbarDataset(Dataset):

    def __init__(self, split_dir, max_images=None, train=False):
        self.paths = sorted(Path(split_dir).glob("*.npz"))
        if not self.paths:
            raise FileNotFoundError(f"No .npz files found in {split_dir}")
        if max_images is not None:
            self.paths = self.paths[:max_images]
        self.train = train

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        with np.load(self.paths[idx]) as data:
            image = data["image"]   # (D, H, W)
            label = data["label"]   # (D, H, W)

        if self.train:
            image, label = self._random_crop(image, label)
        else:
            image, label = self._center_crop(image, label)

        image = torch.from_numpy(image.copy()).unsqueeze(0).float()  # (1, D, H, W)
        label = torch.from_numpy(label.copy()).long()                # (D, H, W)
        return image, label, self.paths[idx].stem

    def _random_crop(self, image, label):
        pd, ph, pw = PATCH_SIZE
        d, h, w = image.shape

        fg = np.argwhere(label > 0)
        if len(fg) > 0:
            center = fg[np.random.randint(len(fg))]
            center = center + np.random.randint(-16, 17, size=3)
        else:
            center = np.array([d // 2, h // 2, w // 2])

        sz = int(np.clip(center[0] - pd // 2, 0, max(d - pd, 0)))
        sy = int(np.clip(center[1] - ph // 2, 0, max(h - ph, 0)))
        sx = int(np.clip(center[2] - pw // 2, 0, max(w - pw, 0)))

        patch_img = image[sz:sz+pd, sy:sy+ph, sx:sx+pw]
        patch_lbl = label[sz:sz+pd, sy:sy+ph, sx:sx+pw]

        # pad if volume is smaller than patch in any dimension
        pad_d = pd - patch_img.shape[0]
        pad_h = ph - patch_img.shape[1]
        pad_w = pw - patch_img.shape[2]
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            patch_img = np.pad(patch_img, ((0, max(pad_d,0)), (0, max(pad_h,0)), (0, max(pad_w,0))))
            patch_lbl = np.pad(patch_lbl, ((0, max(pad_d,0)), (0, max(pad_h,0)), (0, max(pad_w,0))))

        return patch_img, patch_lbl

    def _center_crop(self, image, label):
        pd, ph, pw = PATCH_SIZE
        d, h, w = image.shape

        sz = max(0, (d - pd) // 2)
        sy = max(0, (h - ph) // 2)
        sx = max(0, (w - pw) // 2)

        patch_img = image[sz:sz+pd, sy:sy+ph, sx:sx+pw]
        patch_lbl = label[sz:sz+pd, sy:sy+ph, sx:sx+pw]

        pad_d = pd - patch_img.shape[0]
        pad_h = ph - patch_img.shape[1]
        pad_w = pw - patch_img.shape[2]
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            patch_img = np.pad(patch_img, ((0, max(pad_d,0)), (0, max(pad_h,0)), (0, max(pad_w,0))))
            patch_lbl = np.pad(patch_lbl, ((0, max(pad_d,0)), (0, max(pad_h,0)), (0, max(pad_w,0))))

        return patch_img, patch_lbl


def get_dataloader(split_dir, batch_size=1, shuffle=False, num_workers=2, max_images=None, train=False):
    dataset = LumbarDataset(split_dir, max_images=max_images, train=train)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


if __name__ == "__main__":
    loader = get_dataloader("../processed/train", batch_size=1, train=True)
    image, label, case_id = next(iter(loader))
    print(f"case: {case_id[0]}")
    print(f"image: {image.shape} {image.dtype}")
    print(f"label: {label.shape} {label.dtype}  classes: {label.unique().tolist()}")
