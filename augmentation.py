import numpy as np
import torch as tc
import torchio as tcio

def generate_transforms_flip_affine(scales=(0.98, 1.02), degrees=(-5, 5), translation=(-3, 3)):
    """
    Simple transformations generator (affine) to augment the dataset.
    """
    transforms = tcio.transforms.Compose([
        tcio.RandomFlip(),
        tcio.RandomAffine(scales, degrees, translation, image_interpolation="nearest")
    ])
    return transforms

def generate_transforms_affine(scales=(0.98, 1.02), degrees=(-5, 5), translation=(-3, 3)):
    """
    Simple transformations generator (affine) to augment the dataset.
    """
    transforms = tcio.transforms.Compose([
        tcio.RandomAffine(scales, degrees, translation, image_interpolation="nearest")
    ])
    return transforms

def apply_transform(first_image, second_image=None, transforms=generate_transforms_affine()):
    """
    Apply the given transformation generator to the images.
    """
    if second_image is None:
        return transforms(first_image[np.newaxis, :, :, :])[0]
    else:
        composed = np.stack((first_image, second_image), axis=0)
        transformed = transforms(composed)
        return transformed[0], transformed[1]