from typing import Callable

import torch
from torchvision.transforms import transforms as tfs

from lipschitz.data.preprocessors import convert
from lipschitz.io_functions.key_not_found_error import KeyNotFoundError

AugmentationFunction = Callable[[torch.Tensor], torch.Tensor]


def concatenate(*args):
    return tfs.Compose(args)


def augmentation94percent(h=32, w=32, crop_size=4, erase_proportion=1 / 16):
    crop = tfs.RandomCrop((h, w), padding=crop_size, padding_mode="reflect")
    flip = tfs.RandomHorizontalFlip()
    scale = (erase_proportion, erase_proportion)
    erase = tfs.RandomErasing(p=1., scale=scale, ratio=(1., 1.))
    return tfs.Compose([crop, flip, erase])


def crop_flip_erase_color(h=32, w=32, crop_size=4, erase_proportion=1 / 16,
                          **kwargs):
    aug94 = augmentation94percent(h, w, crop_size, erase_proportion)
    color = color_jitter(**kwargs)
    return tfs.Compose([aug94, color])


def color_crop_flip_erase(h=32,
                          w=32,
                          crop_size=4,
                          erase_proportion=1 / 16,
                          **kwargs):
    color = color_jitter(**kwargs)

    cifar_mean = torch.tensor([0.491, 0.482, 0.447])[:, None, None]

    def center(x):
        return x - cifar_mean.to(x.device)

    def un_center(x):
        return x + cifar_mean.to(x.device)

    aug94 = augmentation94percent(h, w, crop_size, erase_proportion)
    return tfs.Compose([un_center, color, center, aug94])


def color_jitter(b=0.2, c=0., s=0.1, h=0.):
    # {"b": 0.2, "c": 0., "s": 0.1, "h":0.}
    return tfs.ColorJitter(
        brightness=b,
        contrast=c,
        saturation=s,
        hue=h,
    )


def random_crop(h=32, w=32, padding=4, padding_mode="reflect"):
    return tfs.RandomCrop((h, w), padding=padding, padding_mode=padding_mode)


def mnist_crop(h=28, w=28, padding=2, padding_mode="constant"):
    return tfs.RandomCrop((h, w), padding=padding, padding_mode=padding_mode)


class GaussianNoise:
    def __init__(self, sd):
        self.sd = sd

    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.sd


def gaussian_94percent(sd=1 / 8, **kwargs):
    return concatenate(
        augmentation94percent(**kwargs),
        GaussianNoise(sd=sd),
    )


class Convert:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return convert(tensor, **self.kwargs)


AUGMENTATION_FUNCTIONS: dict[str, type[AugmentationFunction]] = {
    "None": lambda: tfs.Compose([]),
    "94percent": augmentation94percent,
    "flip": tfs.RandomHorizontalFlip,
    "94percent_gaussian": gaussian_94percent,
    "crop": random_crop,
    "mnist_crop": mnist_crop,
    "con94percent": lambda: tfs.Compose([Convert(), augmentation94percent()]),
    "crop_flip_erase_color": crop_flip_erase_color,
    "color_jitter": color_jitter,
    "color_crop_flip_erase": color_crop_flip_erase,
}

NAMES = list(AUGMENTATION_FUNCTIONS.keys())
AUGMENTATION = {name: func() for name, func in AUGMENTATION_FUNCTIONS.items()}


def get(name, **kwargs) -> tfs.Compose:
    try:
        aug_cls = AUGMENTATION_FUNCTIONS[name]
    except KeyError:
        raise KeyNotFoundError(name, NAMES, "Augmentation function")
    return aug_cls(**kwargs)
