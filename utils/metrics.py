"""Binary change-detection metrics used during validation."""

import numpy as np
import torch


def eval_metrics(output, target, num_classes=2):
    prediction = output.argmax(dim=1)
    labeled = target.ge(0)
    correct = prediction.eq(target).logical_and(labeled).sum().item()
    labeled_count = labeled.sum().item()

    intersection = prediction[prediction.eq(target)]
    area_inter = torch.bincount(intersection, minlength=num_classes)[:num_classes]
    area_pred = torch.bincount(prediction[labeled], minlength=num_classes)[:num_classes]
    area_label = torch.bincount(target[labeled], minlength=num_classes)[:num_classes]
    area_union = area_pred + area_label - area_inter

    positive = prediction.eq(1)
    changed = target.eq(1)
    tp = positive.logical_and(changed).sum().item()
    fp = positive.logical_and(~changed).sum().item()
    tn = (~positive).logical_and(~changed).sum().item()
    fn = (~positive).logical_and(changed).sum().item()
    return [
        correct,
        labeled_count,
        area_inter.cpu().numpy().astype(np.float64),
        area_union.cpu().numpy().astype(np.float64),
        tp,
        fp,
        tn,
        fn,
    ]
