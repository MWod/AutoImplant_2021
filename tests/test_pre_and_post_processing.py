import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch as tc
import matplotlib.pyplot as plt

import utils as u

def test_find_boundaries_1():
    y_size, x_size, z_size = 167, 198, 144
    b_x, e_x = 50, 90
    b_y, e_y = 0, 67
    b_z, e_z = 111, 144
    image = np.zeros((y_size, x_size, z_size))
    image[b_y:e_y, b_x:e_x, b_z:e_z] = 1
    boundaries = u.find_boundaries(image)
    print("Boundaries: ", boundaries)

def test_crop_and_pad_and_reverse_1():
    y_size, x_size, z_size = 167, 198, 144
    b_x, e_x = 50, 90
    b_y, e_y = 0, 67
    b_z, e_z = 111, 144
    image = np.zeros((y_size, x_size, z_size))
    image[b_y:e_y, b_x:e_x, b_z:e_z] = 1

    cropped_image, to_pad = u.crop_and_pad(image, pad_size=3, offset=9)
    reversed_image = u.reverse_crop_and_pad(cropped_image, to_pad, pad_size=3)

    print("Image shape: ", image.shape)
    print("Cropped shape: ", cropped_image.shape)
    print("Reversed shape: ", reversed_image.shape)

    print("MSE: ", np.mean(np.sqrt((image - reversed_image)**2)))
    print("Is equal?: ", np.all(image == reversed_image))

    z_slice = 120
    plt.figure()
    plt.imshow(image[:, :, z_slice], cmap='gray')

    plt.figure()
    plt.imshow(reversed_image[:, :, z_slice], cmap='gray')

    plt.show()

def test_crop_pad_downsample_and_reverse_1():
    y_size, x_size, z_size = 167, 198, 144
    b_x, e_x = 50, 90
    b_y, e_y = 0, 67
    b_z, e_z = 111, 144
    image = np.zeros((y_size, x_size, z_size))
    image[b_y:e_y, b_x:e_x, b_z:e_z] = 1
    spacing = (1.0, 1.0, 1.0)
    new_spacing = (2.0, 2.0, 2.0)

    processed_image, to_pad, internal_shape = u.crop_pad_and_downsample(image, spacing, new_spacing, pad_size=5, offset=10)
    reversed_image = u.reverse_crop_pad_and_downsample(processed_image, spacing, new_spacing, to_pad, internal_shape, pad_size=5)

    print("Image shape: ", image.shape)
    print("Processed shape: ", processed_image.shape)
    print("Reversed shape: ", reversed_image.shape)

    print("MSE: ", np.mean(np.sqrt((image - reversed_image)**2)))
    print("Is equal?: ", np.all(image == reversed_image))

    z_slice = 120
    plt.figure()
    plt.imshow(image[:, :, z_slice], cmap='gray')

    plt.figure()
    plt.imshow(reversed_image[:, :, z_slice], cmap='gray')

    plt.figure()
    plt.imshow(np.abs(reversed_image[:, :, z_slice] - image[:, :, z_slice]), cmap='gray')

    plt.show()

def test_preprocess_postprocess_1():
    y_size, x_size, z_size = 167, 198, 144
    b_x, e_x = 50, 90
    b_y, e_y = 0, 67
    b_z, e_z = 111, 144
    image = np.zeros((y_size, x_size, z_size))
    image[b_y:e_y, b_x:e_x, b_z:e_z] = 1
    spacing = (1.0, 1.0, 1.0)
    new_spacing = (2.0, 2.0, 2.0)
    offset = 8
    pad_size = 5
    output_size = (230, 230, 230)

    processed_image, to_pad, internal_shape, padding = u.preprocess_case(image, spacing, new_spacing, pad_size=pad_size, output_size=output_size, offset=offset)
    print("Image shape: ", image.shape)
    print("Processed shape: ", processed_image.shape)
    reversed_image = u.postprocess_case(processed_image, spacing, new_spacing, padding, to_pad, internal_shape, pad_size)
    print("Reversed shape: ", reversed_image.shape)

    print("MSE: ", np.mean(np.sqrt((image - reversed_image)**2)))
    print("Is equal?: ", np.all(image == reversed_image))

    z_slice = 120
    plt.figure()
    plt.imshow(image[:, :, z_slice], cmap='gray')

    plt.figure()
    plt.imshow(processed_image[:, :, 10], cmap='gray')

    plt.figure()
    plt.imshow(reversed_image[:, :, z_slice], cmap='gray')

    plt.figure()
    plt.imshow(np.abs(reversed_image[:, :, z_slice] - image[:, :, z_slice]), cmap='gray')

    plt.show()




def run():
    # test_find_boundaries_1()
    test_crop_and_pad_and_reverse_1()
    test_crop_pad_downsample_and_reverse_1()
    test_preprocess_postprocess_1()
    pass


if __name__ == "__main__":
    run()