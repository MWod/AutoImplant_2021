import io
import pathlib
import math

import numpy as np
import torch as tc
import matplotlib
import matplotlib.pyplot as plt
import SimpleITK as sitk
from scipy import ndimage as nd
from skimage import measure
from torch.functional import _return_counts


def load_volume(input_path : pathlib.Path, load_origin : bool=False, load_direction : bool=False):
    """
    Utility function used to load 3-D NRRD volume.
    """
    image = sitk.ReadImage(str(input_path))
    spacing = image.GetSpacing()
    volume = sitk.GetArrayFromImage(image).swapaxes(0, 1).swapaxes(1, 2)
    metadata = dict()
    if load_origin:
        origin = image.GetOrigin()
        metadata['origin'] = origin
    if load_direction:
        direction = image.GetDirection()
        metadata['direction'] = direction
    return volume, spacing, metadata

def save_volume(array : np.ndarray, spacing : tuple, save_path : pathlib.Path, use_compression : bool=True, origin : tuple=None, direction : tuple=None):
    """
    Utility function used to save 3-D NRRD volume.
    """
    image  = sitk.GetImageFromArray(array.swapaxes(2, 1).swapaxes(1, 0).astype(np.uint8))
    image.SetSpacing(spacing)
    if origin is not None:
        image.SetOrigin(origin)
    if direction is not None:
        image.SetDirection(direction)
    sitk.WriteImage(image, str(save_path), useCompression=use_compression)

def load_training_case(complete_skull_path : pathlib.Path, defective_skull_path : pathlib.Path, implant_path : pathlib.Path):
    """
    Utility function used to load a given training case.
    """
    complete_skull, spacing, _ = load_volume(complete_skull_path)
    defective_skull, _, _ = load_volume(defective_skull_path)
    implant, _, _ = load_volume(implant_path)
    return complete_skull, defective_skull, implant, spacing

def load_testing_case(defective_skull_path : pathlib.Path):
    """
    Utility function used to load a given testing case.
    """
    defective_skull, spacing, _ = load_volume(defective_skull_path)
    return defective_skull, spacing

def show_training_case(complete_skull : np.ndarray , defective_skull : np.ndarray, implant: np.ndarray, spacing : tuple,
    x_slice=None, y_slice=None, z_slice=None,
    names: list=["Defective", "Complete", "Implant"],
    show: bool=True, return_buffer: bool=True, suptitle: str=None
    ):
    """
    Utility function used to inspect a given training case.
    """
    y_size, x_size, z_size = defective_skull.shape
    x_slice = int((x_size - 1) / 2) if x_slice is None else x_slice
    y_slice = int((y_size - 1) / 2) if y_slice is None else y_slice
    z_slice = int((z_size - 1) / 2) if z_slice is None else z_slice

    fig = plt.figure(dpi=200)
    font = {'size' : 8}
    matplotlib.rc('font', **font)

    show_complete = True if complete_skull is not None else False
    show_implant = True if implant is not None else False

    if show_complete and show_implant:
        rows, cols = 3, 3
    elif show_complete:
        rows, cols = 3, 2
    elif show_implant:
        rows, cols = 3, 2
    else:
        rows, cols = 1, 3

    ### Z Axis
    current_image = 1
    ax = fig.add_subplot(rows, cols, current_image)
    ax.imshow(np.flip(defective_skull[:, :, z_slice], axis=1), cmap='gray')
    ax.axis('off')
    ax.set_title(names[0] + " Z Slice: " + str(z_slice))

    if show_complete:
        current_image += 1
        ax = fig.add_subplot(rows, cols, current_image)
        ax.imshow(np.flip(complete_skull[:, :, z_slice], axis=1), cmap='gray')
        ax.axis('off')
        ax.set_title(names[1] + " Z Slice: " + str(z_slice))

    if show_implant:
        current_image += 1
        ax = fig.add_subplot(rows, cols, current_image)
        ax.imshow(np.flip(implant[:, :, z_slice], axis=1), cmap='gray')
        ax.axis('off')
        ax.set_title(names[2] + " Z Slice: " + str(z_slice))

    ### Y Axis
    current_image += 1
    ax = fig.add_subplot(rows, cols, current_image)
    ax.imshow(np.flip(defective_skull[y_slice, :, :].T), cmap='gray')
    ax.axis('off')
    ax.set_title(names[0] + " Y Slice: " + str(y_slice))

    if show_complete:
        current_image += 1
        ax = fig.add_subplot(rows, cols, current_image)
        ax.imshow(np.flip(complete_skull[y_slice, :, :].T), cmap='gray')
        ax.axis('off')
        ax.set_title(names[1] + " Y Slice: " + str(y_slice))

    if show_implant:
        current_image += 1
        ax = fig.add_subplot(rows, cols, current_image)
        ax.imshow(np.flip(implant[y_slice, :, :].T), cmap='gray')
        ax.axis('off')
        ax.set_title(names[2] + " Y Slice: " + str(y_slice))

    ### X Axis
    current_image += 1
    ax = fig.add_subplot(rows, cols, current_image)
    ax.imshow(np.flip(defective_skull[:, x_slice, :].T), cmap='gray')
    ax.axis('off')
    ax.set_title(names[0] + " X Slice: " + str(x_slice))

    if show_complete:
        current_image += 1
        ax = fig.add_subplot(rows, cols, current_image)
        ax.imshow(np.flip(complete_skull[:, x_slice, :].T), cmap='gray')
        ax.axis('off')
        ax.set_title(names[1] + " X Slice: " + str(x_slice))

    if show_implant:
        current_image += 1
        ax = fig.add_subplot(rows, cols, current_image)
        ax.imshow(np.flip(implant[:, x_slice, :].T), cmap='gray')
        ax.axis('off')
        ax.set_title(names[2] + " X Slice: " + str(x_slice))

    if suptitle is not None:
        plt.suptitle(suptitle)
    
    if show:
        plt.show()

    if return_buffer:
        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg')
        buf.seek(0)
        return buf

def image_warping(image: np.ndarray, displacement_field: np.ndarray, order: int=1, cval: float=0.0):
    """
    Warps the given image using the provided displacement field.
    """
    grid_x, grid_y, grid_z = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]), np.arange(image.shape[2]))
    transformed_image = nd.map_coordinates(image, [grid_y + displacement_field[1], grid_x + displacement_field[0], grid_z + displacement_field[2]], order=order, cval=cval)
    return transformed_image

### Utility functions to perform the preprocessing/postprocessing below ###

def find_boundaries(image : np.ndarray, offset : int=0):
    """
    Utility function to find the skull boundaries.
    """
    x_indices = np.where(np.any(image, axis=(0, 2)))[0]
    y_indices = np.where(np.any(image, axis=(1, 2)))[0]
    z_indices = np.where(np.any(image, axis=(0, 1)))[0]
    b_x, e_x = x_indices[0], x_indices[-1] + 1
    b_y, e_y = y_indices[0], y_indices[-1] + 1
    b_z, e_z = z_indices[0], z_indices[-1] + 1
    if offset > 0:
        y_size, x_size, z_size = image.shape
        b_x, b_y, b_z = b_x - offset, b_y - offset, b_z - offset
        e_x, e_y, e_z = e_x + offset, e_y + offset, e_z + offset
        b_x, b_y, b_z = max(b_x, 0), max(b_y, 0), max(b_z, 0)
        e_x, e_y, e_z = min(e_x, x_size), min(e_y, y_size), min(e_z, z_size)
    boundaries = b_y, e_y, b_x, e_x, b_z, e_z
    return boundaries

def resample_to_shape(image : np.ndarray, new_shape: tuple):
    """
    Resamples image to a given shape.
    """
    shape = image.shape
    grid_x, grid_y, grid_z = np.meshgrid(np.arange(new_shape[1]), np.arange(new_shape[0]), np.arange(new_shape[2]))
    grid_x = grid_x * (shape[1] / new_shape[1])
    grid_y = grid_y * (shape[0] / new_shape[0])
    grid_z = grid_z * (shape[2] / new_shape[2])
    resampled_image = nd.map_coordinates(image, [grid_y, grid_x, grid_z], order=0, cval=0)
    return resampled_image

def resample_to_spacing(image : np.ndarray, old_spacing : tuple, new_spacing : tuple):
    """
    Resamples image to a given spacing.
    """
    shape = image.shape
    multiplier = (np.array(old_spacing, dtype=np.float32) / np.array(new_spacing, dtype=np.float32))
    multiplier[0], multiplier[1] = multiplier[1], multiplier[0]
    new_shape = shape * multiplier
    new_shape = np.ceil(new_shape).astype(np.int)
    resampled_image = resample_to_shape(image, new_shape)
    return resampled_image

def crop_and_pad(image : np.ndarray, boundaries: tuple=None, pad_size: int=0, offset: int=0):
    """
    Utility function to crop the input image to skull only and pad the boundaries.
    """
    y_size, x_size, z_size = image.shape
    if boundaries is None:
        boundaries = find_boundaries(image, offset=offset)
    b_y, e_y, b_x, e_x, b_z, e_z = boundaries
    cropped_image = image[b_y:e_y, b_x:e_x, b_z:e_z]
    to_pad = (b_y, y_size-e_y, b_x, x_size-e_x, b_z, z_size-e_z)
    padded_image = np.pad(cropped_image, ((pad_size, pad_size), (pad_size, pad_size), (pad_size, pad_size)), mode='constant', constant_values=0)
    return padded_image, to_pad

def reverse_crop_and_pad(image : np.ndarray, to_pad, pad_size : int=5):
    """
    Utility function to reverse the crop_and_pad operation.
    """
    unpadded_image = image[pad_size:-pad_size, pad_size:-pad_size, pad_size:-pad_size]
    b_y, e_y, b_x, e_x, b_z, e_z = to_pad
    unpadded_image = np.pad(unpadded_image, ((b_y, e_y), (b_x, e_x), (b_z, e_z)), mode="constant", constant_values=0)
    return unpadded_image

def crop_pad_and_downsample(image : np.ndarray, spacing : tuple, new_spacing: tuple, boundaries : tuple=None, pad_size : int=0, offset : int=0):
    """
    Utility function to boundary crop and downsample the given image.
    """
    crop_pad_image, to_pad = crop_and_pad(image, boundaries=boundaries, pad_size=pad_size, offset=offset)
    resampled_image = resample_to_spacing(crop_pad_image, spacing, new_spacing)
    return resampled_image, to_pad, crop_pad_image.shape

def reverse_crop_pad_and_downsample(image : np.ndarray, spacing : tuple, new_spacing : tuple, to_pad, internal_shape, pad_size=5):
    """
    Utility function to reverse the crop, pad, and resample process.
    """
    resampled_image = resample_to_shape(image, internal_shape)
    reversed_image = reverse_crop_and_pad(resampled_image, to_pad, pad_size=pad_size)
    return reversed_image

def pad_to_shape(image : np.ndarray, output_shape : tuple):
    """
    Utility function to pad the input image to given shape
    """
    y_size, x_size, z_size = image.shape
    new_y_size, new_x_size, new_z_size = output_shape

    def calc_padding(size, new_size):
        diff = new_size - size
        bp, ep = (int(math.floor(diff / 2)), int(math.ceil(diff / 2))) if diff % 2 else (diff // 2, diff // 2)
        return bp, ep

    b_y, e_y = calc_padding(y_size, new_y_size)
    b_x, e_x = calc_padding(x_size, new_x_size)
    b_z, e_z = calc_padding(z_size, new_z_size)

    padded_image = np.pad(image, ((b_y, e_y), (b_x, e_x), (b_z, e_z)), mode="constant", constant_values=0)
    return padded_image, (b_x, e_x, b_y, e_y, b_z, e_z)

def preprocess_case(image : np.ndarray, spacing : tuple, new_spacing : tuple, output_size: tuple, boundaries : tuple=None, pad_size: int=5, offset: int=0):
    """
    Utility function to preprocess a given case.
    """
    preprocessed_image, to_pad, internal_shape = crop_pad_and_downsample(image, spacing, new_spacing, boundaries=boundaries, pad_size=pad_size, offset=offset)
    preprocessed_image, padding = pad_to_shape(preprocessed_image, output_size)
    return preprocessed_image, to_pad, internal_shape, padding

def postprocess_case(image : np.ndarray, spacing : tuple, new_spacing : tuple, padding: tuple, to_pad : int, internal_shape : tuple, pad_size : int=5):
    """
    Utility function to postprocess a single case.
    """
    y_size, x_size, z_size = image.shape
    b_x, e_x, b_y, e_y, b_z, e_z = padding
    postprocessed_image = image[b_y:y_size-e_y, b_x:x_size-e_x, b_z:z_size-e_z]
    postprocessed_image = reverse_crop_pad_and_downsample(postprocessed_image, spacing, new_spacing, to_pad, internal_shape, pad_size=pad_size)
    return postprocessed_image

def binary_postprocessing(reconstructed_defect : np.ndarray, defective_skull : np.ndarray):
    """
    Simple binary postprocessing basic on morphology operations to improve the defect boundaries
    """
    complete_skull = np.logical_or(reconstructed_defect, defective_skull)
    closed_skull = nd.morphology.binary_closing(complete_skull)
    improved_defect = np.logical_xor(closed_skull, defective_skull)
    labels = measure.label(improved_defect)
    unique, counts = np.unique(labels, return_counts=True)
    counts[np.argmax(counts)] = 0 # Delete background
    improved_defect[labels != unique[np.argmax(counts)]] = 0
    return improved_defect

def preprocess_training_case(defective_skull : np.ndarray, complete_skull : np.ndarray, implant : np.ndarray,
    spacing : tuple, output_spacing : tuple, pad_size : int, output_size : tuple, offset : int):
    """
    Function to preprocess a given training case.
    """
    boundaries = find_boundaries(complete_skull)
    preprocessed_defective_skull, to_pad, internal_shape, padding = preprocess_case(defective_skull, spacing, output_spacing, boundaries=boundaries, pad_size=pad_size, output_size=output_size, offset=offset)
    preprocessed_complete_skull, _, _, _ = preprocess_case(complete_skull, spacing, output_spacing, boundaries=boundaries, pad_size=pad_size, output_size=output_size, offset=offset)
    preprocessed_implant, _, _, _ = preprocess_case(implant, spacing, output_spacing, boundaries=boundaries, pad_size=pad_size, output_size=output_size, offset=offset)
    return preprocessed_complete_skull, preprocessed_defective_skull, preprocessed_implant, to_pad, internal_shape, padding

def preprocess_testing_case(defective_skull: np.ndarray, spacing : tuple, output_spacing : tuple, pad_size : int, output_size : tuple, offset : int):
    """
    Function to preprocess a given testing case.
    """
    boundaries = find_boundaries(defective_skull)
    preprocessed_defective_skull, to_pad, internal_shape, padding = preprocess_case(defective_skull, spacing, output_spacing, boundaries=boundaries, pad_size=pad_size, output_size=output_size, offset=offset)
    return preprocessed_defective_skull, to_pad, internal_shape, padding

def dice_loss(prediction, target):
    """
    Dice as PyTorch cost function.
    """
    smooth = 1
    prediction = prediction.contiguous().view(-1)
    target = target.contiguous().view(-1)
    intersection = tc.sum(prediction * target)
    return 1 - ((2 * intersection + smooth) / (prediction.sum() + target.sum() + smooth))

def dice_loss_multichannel(prediction, target):
    """
    Dice loss for multichannel masks (equally averaged)
    """
    no_channels = prediction.size(1)
    for i in range(no_channels):
        if i == 0:
            loss = dice_loss(prediction[:, i, :, :, :], target[:, i, :, :, :])
        else:
            loss += dice_loss(prediction[:, i, :, :, :], target[:, i, :, :, :])
    loss = loss / no_channels
    return loss

def kld_approx(z, mean, std):
    """
    Approximate KL divergence (only for big batches)
    """
    p = tc.distributions.Normal(tc.zeros_like(mean), tc.ones_like(std))
    q = tc.distributions.Normal(mean, std)
    log_qzx = q.log_prob(z)
    log_pz = p.log_prob(z)
    kl = (log_qzx - log_pz)
    kl = kl.sum(-1)
    return tc.mean(kl)

def kld(z, mean, std):
    """
    Direct KL divergence (for all-sized batches)
    """
    p = tc.distributions.Normal(tc.zeros_like(mean), tc.ones_like(std))
    q = tc.distributions.Normal(mean, std)
    kl = tc.distributions.kl_divergence(q, p)
    kl = kl.mean()
    return kl 

def gl(x_hat, logscale, x):
    """
    Gaussian-likelihood
    """
    scale = tc.exp(logscale)
    mean = x_hat
    dist = tc.distributions.Normal(mean, scale)
    log_pxz = dist.log_prob(x)
    return -log_pxz.mean(dim=(1, 2, 3, 4))

def parse_state_dict(initial_weights_path, keyword='unet.'):
    """
    Function to parse the state of PyTorch Lightning module directly to PyTorch model.
    """
    state_dict = tc.load(initial_weights_path)
    output_state_dict = dict()
    for key, value in state_dict.items():
        output_state_dict[key.replace(keyword, "")] = value
    return output_state_dict

def defect_refinement(implant, model, output_shape, boundary_offset, device="cpu"):
    """
    Function to perform the defect refinement.
    """
    refinement_boundaries = find_boundaries(implant, offset=boundary_offset)
    b_y, e_y, b_x, e_x, b_z, e_z = refinement_boundaries
    input_to_implant_refinement = implant[b_y:e_y, b_x:e_x, b_z:e_z]
    original_refinement_shape = input_to_implant_refinement.shape
    input_to_implant_refinement = resample_to_shape(input_to_implant_refinement, output_shape)
    input_to_implant_refinement = tc.from_numpy(input_to_implant_refinement.astype(np.float32)).view(1, 1, *input_to_implant_refinement.shape).to(device)
    with tc.set_grad_enabled(False):
        refined_defective_skull = model(input_to_implant_refinement)
    refined_defective_skull = refined_defective_skull[0, 0, :, :, :].detach().cpu().numpy() > 0.5
    refined_defective_skull = resample_to_shape(refined_defective_skull, original_refinement_shape)
    implant[b_y:e_y, b_x:e_x, b_z:e_z] = refined_defective_skull
    return implant