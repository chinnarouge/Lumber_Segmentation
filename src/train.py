import argparse
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb

from data_loader import get_dataloader, NUM_CLASSES
from model import UNet3D
from evaluation import evaluate_loader


def dice_loss(logits, target, num_classes, eps=1e-5):
    probs = F.softmax(logits, dim=1)
    target_onehot = F.one_hot(target, num_classes).permute(0, 4, 1, 2, 3).float()
    dims = (0, 2, 3, 4)
    intersection = (probs * target_onehot).sum(dims)
    union = probs.sum(dims) + target_onehot.sum(dims)
    dice_per_class = (2 * intersection + eps) / (union + eps)
    return 1 - dice_per_class[1:].mean()  # exclude background


def combined_loss(logits, target, num_classes):
    return F.cross_entropy(logits, target) + dice_loss(logits, target, num_classes)


def save_checkpoint(model, optimizer, scheduler, epoch, best_dice, run_dir):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "best_dice": best_dice,
    }, run_dir / "checkpoint_last.pt")


def load_checkpoint(run_dir, model, optimizer, scheduler, device):
    path = run_dir / "checkpoint_last.pt"
    if not path.exists():
        return 0, 0.0
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if "scheduler" in ckpt and scheduler is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    print(f"Resumed from epoch {ckpt['epoch']} (best_dice={ckpt['best_dice']:.4f})")
    return ckpt["epoch"], ckpt["best_dice"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_dir", default="processed/train")
    p.add_argument("--val_dir", default="processed/val")
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--lr_patience", type=int, default=5)
    p.add_argument("--lr_factor", type=float, default=0.5)
    p.add_argument("--max_train_images", type=int, default=None)
    p.add_argument("--max_val_images", type=int, default=None)
    p.add_argument("--ts_test_dir", default="processed/test")
    p.add_argument("--verse_test_dir", default="processed/verse_test")
    p.add_argument("--run_name", default=None)
    p.add_argument("--base_channels", type=int, default=32)
    p.add_argument("--wandb_project", default="lumbar-segmentation")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_name = args.run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(args.checkpoint_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    model = UNet3D(in_channels=1, num_classes=NUM_CLASSES, base_channels=args.base_channels).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: UNet3D (base_channels={args.base_channels}, params={params:,})")
    print(f"Device: {device}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=args.lr_patience, factor=args.lr_factor)
    start_epoch, best_dice = load_checkpoint(run_dir, model, optimizer, scheduler, device)

    wandb.init(project=args.wandb_project, name=run_name, id=run_name, config=vars(args), resume="allow")

    train_loader = get_dataloader(args.train_dir, batch_size=args.batch_size, shuffle=True, max_images=args.max_train_images, train=True)
    val_loader = get_dataloader(args.val_dir, batch_size=1, shuffle=False, max_images=args.max_val_images, train=False)

    test_loaders = {}
    for name, test_dir in [("ts_test", args.ts_test_dir), ("verse_test", args.verse_test_dir)]:
        p = Path(test_dir)
        if p.exists() and list(p.glob("*.npz")):
            test_loaders[name] = get_dataloader(test_dir, batch_size=1, shuffle=False)

    print(f"Data loaded: train={len(train_loader.dataset)}, val={len(val_loader.dataset)}, test={{{', '.join(f'{k}={len(v.dataset)}' for k, v in test_loaders.items())}}}")
    print(f"Run: {run_name} | epochs={args.epochs} | lr={args.lr} | patience={args.patience}")
    print(f"Checkpoints: {run_dir}")
    if start_epoch > 0:
        print(f"Resumed from epoch {start_epoch} (best_dice={best_dice:.4f})")
    print("-" * 60)
    print("Training started")

    epochs_no_improve = 0

    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for image, label, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            image, label = image.to(device), label.to(device)
            logits = model(image)
            loss = combined_loss(logits, label, NUM_CLASSES)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        print("Validating...")
        val_dice, _, val_loss = evaluate_loader(model, val_loader, device, NUM_CLASSES)

        scheduler.step(val_dice["mean"])
        lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} lr={lr:.2e} | L1={val_dice['L1']:.4f}  L2={val_dice['L2']:.4f}  L3={val_dice['L3']:.4f}  L4={val_dice['L4']:.4f}  L5={val_dice['L5']:.4f} | mean={val_dice['mean']:.4f}")
        log = {
            "epoch": epoch,
            "loss/train": train_loss,
            "loss/val": val_loss,
            "lr": lr,
            "dice/val_mean": val_dice["mean"],
            **{f"dice/val_{n}": s for n, s in val_dice.items() if n != "mean"},
        }

        for test_name, test_loader in test_loaders.items():
            test_dice, _, test_loss = evaluate_loader(model, test_loader, device, NUM_CLASSES)
            print(f"  {test_name} | loss={test_loss:.4f} | L1={test_dice['L1']:.4f}  L2={test_dice['L2']:.4f}  L3={test_dice['L3']:.4f}  L4={test_dice['L4']:.4f}  L5={test_dice['L5']:.4f} | mean={test_dice['mean']:.4f}")
            log[f"loss/{test_name}"] = test_loss
            log[f"dice/{test_name}_mean"] = test_dice["mean"]
            for v in ["L1", "L2", "L3", "L4", "L5"]:
                log[f"dice/{test_name}_{v}"] = test_dice[v]

        wandb.log(log)

        save_checkpoint(model, optimizer, scheduler, epoch, best_dice, run_dir)

        if val_dice["mean"] > best_dice:
            best_dice = val_dice["mean"]
            torch.save(model.state_dict(), run_dir / "best_model.pt")
            epochs_no_improve = 0
            print(f"  -> new best (dice={best_dice:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  no improvement ({epochs_no_improve}/{args.patience})")

        if epochs_no_improve >= args.patience:
            print(f"Early stopping: no improvement for {args.patience} epochs")
            break

    wandb.finish()
    print("-" * 60)
    print(f"Training complete. Best val dice: {best_dice:.4f}")
    print(f"Best model: {run_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
