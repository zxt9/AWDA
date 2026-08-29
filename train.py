"""Train AWDA with a ResNet-50 backbone."""

import argparse
import json
import random
import numpy as np
import torch

from dataloaders import CDDataset
from models import AWDA
from trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train AWDA for cross-dataset change detection")
    parser.add_argument("--config", default="configs/awda_resnet50.json")
    parser.add_argument("--resume", default=None, help="checkpoint to resume")
    parser.add_argument("--device", default="cuda", help="cuda, cuda:0, or cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, encoding="utf-8") as handle:
        config = json.load(handle)

    seed = config.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    source_loader = CDDataset(dict(config["train_source"]))
    target_loader = CDDataset(dict(config["train_target"]))
    val_loader = CDDataset(dict(config["val_loader"]))
    model = AWDA(num_classes=val_loader.dataset.num_classes,
                 confidence_threshold=config["model"]["confidence_threshold"],
                 pretrained=config["model"].get("pretrained", True))
    trainer = Trainer(model, config, source_loader, target_loader, val_loader,
                      device=args.device, resume=args.resume)
    trainer.train()


if __name__ == "__main__":
    main()
