"""Training and validation loop for AWDA."""

import json
from itertools import cycle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.lr_scheduler import Poly
from utils.metrics import eval_metrics


class Trainer:
    def __init__(self, model, config, source_loader, target_loader, val_loader, device="cuda", resume=None):
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.config = config
        self.source_loader, self.target_loader, self.val_loader = source_loader, target_loader, val_loader
        self.epochs = config["trainer"]["epochs"]
        self.start_epoch, self.best_iou = 0, 0.0
        self.save_dir = Path(config["trainer"]["save_dir"])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        with (self.save_dir / "config.json").open("w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2)

        opt = config["optimizer"]
        params = [
            {"params": self.model.get_other_params()},
            {"params": self.model.get_backbone_params(), "lr": opt["lr"] / 10},
        ]
        self.optimizer = torch.optim.SGD(params, lr=opt["lr"], momentum=opt["momentum"],
                                         weight_decay=opt["weight_decay"])
        iterations = max(len(source_loader), len(target_loader))
        self.scheduler = Poly(self.optimizer, self.epochs, iterations)
        if resume:
            self._load(resume)

    def train(self):
        for epoch in range(self.start_epoch, self.epochs):
            losses = self._train_epoch(epoch)
            change_iou = self._validate()
            improved = change_iou > self.best_iou
            self.best_iou = max(self.best_iou, change_iou)
            self._save(epoch, improved)
            fore, back = self.model.class_weights(epoch, self.epochs)
            print(f"epoch={epoch + 1} loss={losses:.4f} IoU(change)={change_iou:.4f} "
                  f"weights(fore/back)={fore:.3f}/{back:.3f}")

    def _train_epoch(self, epoch):
        self.model.train()
        pairs = (zip(cycle(self.source_loader), self.target_loader)
                 if len(self.source_loader) < len(self.target_loader)
                 else zip(self.source_loader, cycle(self.target_loader)))
        running = 0.0
        bar = tqdm(pairs, total=max(len(self.source_loader), len(self.target_loader)), desc="train")
        for step, (source, target) in enumerate(bar):
            source_a, source_b, source_label = (item.to(self.device, non_blocking=True) for item in source)
            weak_a, weak_b, strong_a, strong_b, _ = (item.to(self.device, non_blocking=True) for item in target)
            self.optimizer.zero_grad(set_to_none=True)
            losses, _ = self.model(epoch=epoch, total_epochs=self.epochs, source_a=source_a,
                source_b=source_b, source_label=source_label, target_weak_a=weak_a,
                target_weak_b=weak_b, target_strong_a=strong_a, target_strong_b=strong_b)
            total = losses["source"] + losses["target"] + losses["domain"]
            total.backward()
            self.optimizer.step()
            self.scheduler.step()
            running += total.item()
            bar.set_postfix(loss=f"{running / (step + 1):.4f}")
        return running / max(len(bar), 1)

    @torch.no_grad()
    def _validate(self):
        self.model.eval()
        intersection = np.zeros(2)
        union = np.zeros(2)
        for image_a, image_b, label, _ in tqdm(self.val_loader, desc="validate"):
            image_a, image_b, label = image_a.to(self.device), image_b.to(self.device), label.to(self.device)
            logits = self.model(image_a=image_a, image_b=image_b)
            logits = F.interpolate(logits, size=label.shape[-2:], mode="bilinear", align_corners=False)
            _, _, batch_inter, batch_union, *_ = eval_metrics(logits, label, 2)
            intersection += batch_inter
            union += batch_union
        return float(intersection[1] / (union[1] + np.finfo(float).eps))

    def _save(self, epoch, best):
        state = {"epoch": epoch, "state_dict": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                 "scheduler": self.scheduler.state_dict(), "best_iou": self.best_iou, "config": self.config}
        torch.save(state, self.save_dir / "last.pth")
        if best:
            torch.save(state, self.save_dir / "best.pth")

    def _load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.start_epoch = checkpoint["epoch"] + 1
        self.best_iou = checkpoint.get("best_iou", 0.0)
