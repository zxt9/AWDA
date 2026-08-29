"""AWDA change detector with a ResNet-50 backbone."""

from itertools import chain
import numpy as np
import torch
import torch.nn.functional as F
from base import BaseModel
from models.decoder import Decoder
from models.encoder import EncoderResNet50


class AWDA(BaseModel):
    """Adversarial and Weighted Domain Adaptation for change detection."""

    def __init__(self, num_classes=2, confidence_threshold=0.95, pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.encoder = EncoderResNet50(pretrained=pretrained)
        self.decoder = Decoder(upscale=8, conv_in_ch=512, num_classes=num_classes)
        self.domain_decoder = Decoder(upscale=8, conv_in_ch=512, num_classes=2)
        self.foreground_label, self.background_label = 1, 0
        self.momentum = 0.99
        self.register_buffer("mean_source_foreground", torch.ones(1))
        self.register_buffer("mean_source_background", torch.ones(1))

    @torch.no_grad()
    def _update_class_confidence(self, probabilities, labels):
        foreground = probabilities[:, self.foreground_label][labels == self.foreground_label]
        background = probabilities[:, self.background_label][labels == self.background_label]
        if foreground.numel():
            self.mean_source_foreground.mul_(self.momentum).add_(foreground.mean(), alpha=1 - self.momentum)
        if background.numel():
            self.mean_source_background.mul_(self.momentum).add_(background.mean(), alpha=1 - self.momentum)

    def class_weights(self, epoch, total_epochs):
        power = 3.0 - 2.0 * epoch / max(float(total_epochs), 1.0)
        return self.mean_source_foreground.pow(-power).item(), self.mean_source_background.pow(-power).item()

    def forward(self, image_a=None, image_b=None, *, epoch=0, total_epochs=50,
                source_a=None, source_b=None, source_label=None,
                target_weak_a=None, target_weak_b=None,
                target_strong_a=None, target_strong_b=None):
        if not self.training:
            return self.decoder(self.encoder(image_a, image_b))

        progress = epoch / max(float(total_epochs), 1.0)
        domain_alpha = 2.0 / (1.0 + np.exp(-10.0 * progress)) - 1.0
        weight_power = 3.0 - 2.0 * progress
        output_size = source_a.shape[-2:]

        source_features = self.encoder(source_a, source_b)
        source_logits = self.decoder(source_features)
        source_probabilities = F.softmax(source_logits, dim=1)
        source_loss = F.cross_entropy(source_logits, source_label)

        weak_features = self.encoder(target_weak_a, target_weak_b)
        weak_logits = self.decoder(weak_features)
        weak_probabilities = F.softmax(weak_logits, dim=1).detach()
        strong_logits = self.decoder(self.encoder(target_strong_a, target_strong_b))

        domain_source = self.domain_decoder(source_features, reverse_alpha=domain_alpha)
        domain_target = self.domain_decoder(weak_features, reverse_alpha=domain_alpha)
        domain_source = F.interpolate(domain_source, size=output_size, mode="bilinear", align_corners=False)
        domain_target = F.interpolate(domain_target, size=output_size, mode="bilinear", align_corners=False)
        domain_loss = F.cross_entropy(domain_source, torch.zeros_like(source_label))
        domain_loss += F.cross_entropy(domain_target, torch.ones_like(source_label))

        feature_size = weak_features.shape[-2:]
        weak_probabilities = F.interpolate(weak_probabilities, size=feature_size, mode="nearest")
        confidence, pseudo_labels = weak_probabilities.max(dim=1)
        confidence_mask = confidence.gt(self.confidence_threshold).float()
        self._update_class_confidence(source_probabilities, source_label)
        pixel_weights = torch.where(
            pseudo_labels.eq(self.foreground_label),
            self.mean_source_foreground.pow(-weight_power),
            self.mean_source_background.pow(-weight_power),
        )
        strong_logits = F.interpolate(strong_logits, size=feature_size, mode="bilinear", align_corners=False)
        target_loss = (F.cross_entropy(strong_logits, pseudo_labels, reduction="none")
                       * confidence_mask * pixel_weights).mean()

        predictions = {
            "source": F.interpolate(source_logits, size=output_size, mode="bilinear", align_corners=False),
            "target": F.interpolate(weak_logits, size=output_size, mode="bilinear", align_corners=False),
        }
        return {"source": source_loss, "target": target_loss, "domain": domain_loss}, predictions

    def get_backbone_params(self):
        return self.encoder.get_backbone_params()

    def get_other_params(self):
        return chain(self.encoder.get_module_params(), self.decoder.parameters(), self.domain_decoder.parameters())


ResNet50_CD = AWDA
