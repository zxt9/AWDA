"""Strong augmentation used for AWDA target-domain self-training."""

import random

import numpy as np
from PIL import ImageDraw, ImageEnhance, ImageOps


PARAMETER_MAX = 10


def _float_parameter(value, maximum):
    return float(value) * maximum / PARAMETER_MAX


def _int_parameter(value, maximum):
    return int(value * maximum / PARAMETER_MAX)


def auto_contrast(image, **_):
    return ImageOps.autocontrast(image)


def brightness(image, value, maximum, bias=0):
    return ImageEnhance.Brightness(image).enhance(_float_parameter(value, maximum) + bias)


def color(image, value, maximum, bias=0):
    return ImageEnhance.Color(image).enhance(_float_parameter(value, maximum) + bias)


def contrast(image, value, maximum, bias=0):
    return ImageEnhance.Contrast(image).enhance(_float_parameter(value, maximum) + bias)


def equalize(image, **_):
    return ImageOps.equalize(image)


def identity(image, **_):
    return image


def posterize(image, value, maximum, bias=0):
    return ImageOps.posterize(image, _int_parameter(value, maximum) + bias)


def sharpness(image, value, maximum, bias=0):
    return ImageEnhance.Sharpness(image).enhance(_float_parameter(value, maximum) + bias)


def solarize(image, value, maximum, bias=0):
    return ImageOps.solarize(image, 256 - (_int_parameter(value, maximum) + bias))


def cutout(image, size):
    width, height = image.size
    x = int(max(0, np.random.uniform(0, width) - size / 2))
    y = int(max(0, np.random.uniform(0, height) - size / 2))
    image = image.copy()
    ImageDraw.Draw(image).rectangle((x, y, min(width, x + size), min(height, y + size)), (127, 127, 127))
    return image


class RandAugmentMC:
    def __init__(self, n=2, m=10):
        self.n = n
        self.m = m
        self.augment_pool = [
            (auto_contrast, None, None),
            (brightness, 0.9, 0.05),
            (color, 0.9, 0.05),
            (contrast, 0.9, 0.05),
            (equalize, None, None),
            (identity, None, None),
            (posterize, 4, 4),
            (sharpness, 0.9, 0.05),
            (solarize, 256, 0),
        ]

    def __call__(self, image):
        for operation, maximum, bias in random.choices(self.augment_pool, k=self.n):
            value = np.random.randint(1, self.m)
            if random.random() < 0.5:
                image = operation(image, value=value, maximum=maximum, bias=bias)
        return cutout(image, 16)
