import numpy as np
import matplotlib.pyplot as plt

import scipy.ndimage as nd
import skimage.measure as measure

import utils as u

def calculate_centroid(defective_skull, reconstruction, contour):
    """
    Calculates the centroids of: complete skull, the defect reconstruction, and the reconstruction contour
    """
    complete_skull = np.logical_or(reconstruction, defective_skull)
    return np.array(nd.center_of_mass(complete_skull)), np.array(nd.center_of_mass(reconstruction)), np.array(nd.center_of_mass(contour))

def calculate_contour(reconstruction):
    """
    Calculates the inner contour of the provided reconstruction
    """
    contour = np.logical_xor(nd.binary_erosion(reconstruction), reconstruction)
    return contour

def calculate_normal(con_centroid, centroid):
    """
    Calculates the vector between two centroids
    """
    vector = con_centroid - centroid
    norm = np.linalg.norm(vector)
    normal = vector / norm
    return normal

def transform_using_normal(image, normal, offset):
    """
    Translates a given binary image across the given normal vector
    """
    y_size, x_size, z_size = image.shape
    x_grid, y_grid, z_grid = np.meshgrid(np.arange(x_size), np.arange(y_size), np.arange(z_size))
    transforemd_image = nd.map_coordinates(image, [y_grid - offset*normal[0], x_grid - offset*normal[1], z_grid - offset*normal[2]], order=0, cval=0)
    return transforemd_image

def calculate_volume_ratio(transformed_image, template_image):
    """
    Calculates the current volume ratio (between two binary masks)
    """
    return np.sum(transformed_image) / np.sum(template_image)

def search_optimal_implant(reconstruction, defective_skull, normal, max_iters=30, step=1, desired_ratio=0.6, echo=False):
    """
    Iterative process searching for an optimal cranial implant.
    """
    transformed_reconstruction = reconstruction.copy()
    for i in range(max_iters):
        transformed_reconstruction = transform_using_normal(transformed_reconstruction, normal, step)
        transformed_reconstruction = np.logical_and(reconstruction, transformed_reconstruction)
        filled_skull = np.logical_or(transformed_reconstruction, defective_skull)
        filled_skull = nd.median_filter(filled_skull, size=3)
        transformed_reconstruction = np.logical_xor(filled_skull, defective_skull)
        labels = measure.label(transformed_reconstruction)
        unique, counts = np.unique(labels, return_counts=True)
        counts[np.argmax(counts)] = 0
        transformed_reconstruction[labels != unique[np.argmax(counts)]] = 0
        current_ratio = calculate_volume_ratio(transformed_reconstruction, reconstruction)
        print(f"Iteration: {i}, Current ratio: {current_ratio}") if echo else None
        if current_ratio < desired_ratio:
            break
    return transformed_reconstruction

def reconstruction_to_implant(reconstruction, defective_skull, spacing=(1.0, 1.0, 1.0), desired_ratio=0.6, echo=False):
    """
    Function performing simple implant modeling using a given defect reconstruction. It assumes a correct reconstruction.
    """
    contour = calculate_contour(reconstruction)
    centroid, rec_centroid, con_centroid = calculate_centroid(defective_skull, reconstruction, contour)
    normal = calculate_normal(con_centroid, centroid)
    transformed_reconstruction = search_optimal_implant(reconstruction, defective_skull, normal, desired_ratio=desired_ratio, echo=echo)
    return transformed_reconstruction