import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


import numpy as np
import matplotlib.pyplot as plt
import torch as tc

import datasets as ds
import augmentation as aug
import utils as u
import paths as p

def test_task_1_dataloader_1():
    """
    Test whether the dataloader works for Task 1 training cases.
    """
    data_folder = p.task_1_training_preprocessed_path
    csv_file = p.task_1_training_csv_path
    mode = "defect_implant"

    batch_size = 4
    dataloader = ds.create_dataloader(data_folder, csv_file, mode, batch_size)
    print("Dataloader size: ", len(dataloader.dataset))

    for defects, implants, spacings in dataloader:
        print("Defects size: ", defects.size())
        print("Implants size: ", implants.size())
        print("Spacings size: ", spacings.size())

        defect, implant, spacing = defects[0, :, :, :], implants[0, :, :, :], spacings[0]

        print("Defect size: ", defect.size())
        print("Implant size: ", implant.size())
        print("Spacing: ", spacing)

        u.show_training_case(None, defect.numpy(), implant.numpy(), spacing.numpy())

def test_task_1_dataloader_2():
    """
    Test whether the dataloader works for Task 1 evaluation cases.
    """
    data_folder = p.task_1_testing_preprocessed_path
    csv_file = p.task_1_testing_csv_path
    mode = "defect"

    batch_size = 4
    dataloader = ds.create_dataloader(data_folder, csv_file, mode, batch_size)
    print("Dataloader size: ", len(dataloader.dataset))

    for defects, spacings in dataloader:
        print("Defects size: ", defects.size())
        print("Spacings size: ", spacings.size())

        defect, spacing = defects[0, :, :, :], spacings[0]

        print("Defect size: ", defect.size())
        print("Spacing: ", spacing)

        u.show_training_case(None, defect.numpy(), None, spacing.numpy())

def test_combined_dataloader_1():
    """
    Test whether the dataloader works for Combined training cases.
    """
    data_folder = p.combined_training_path
    csv_file = p.combined_training_csv_path
    mode = "defect_implant"

    batch_size = 4
    dataloader = ds.create_dataloader(data_folder, csv_file, mode, batch_size)
    print("Dataloader size: ", len(dataloader.dataset))

    for defects, implants, spacings in dataloader:
        print("Defects size: ", defects.size())
        print("Implants size: ", implants.size())
        print("Spacings size: ", spacings.size())

        defect, implant, spacing = defects[0, :, :, :], implants[0, :, :, :], spacings[0]

        print("Defect size: ", defect.size())
        print("Implant size: ", implant.size())
        print("Spacing: ", spacing)

        u.show_training_case(None, defect.numpy(), implant.numpy(), spacing.numpy())
        break

def test_combined_dataloader_2():
    """
    Test whether the dataloader works for Combined training cases (w/o registration).
    """
    data_folder = p.combined_training_path
    csv_file = p.combined_training_wr_csv_path
    mode = "defect_implant"

    batch_size = 4
    dataloader = ds.create_dataloader(data_folder, csv_file, mode, batch_size)
    print("Dataloader size: ", len(dataloader.dataset))

    for defects, implants, spacings in dataloader:
        print("Defects size: ", defects.size())
        print("Implants size: ", implants.size())
        print("Spacings size: ", spacings.size())

        defect, implant, spacing = defects[0, :, :, :], implants[0, :, :, :], spacings[0]

        print("Defect size: ", defect.size())
        print("Implant size: ", implant.size())
        print("Spacing: ", spacing)

        u.show_training_case(None, defect.numpy(), implant.numpy(), spacing.numpy())
        break

def test_combined_dataloader_3():
    """
    Test whether the dataloader works for Combined training cases (with augmentation).
    """
    data_folder = p.combined_training_path
    csv_file = p.combined_training_csv_path
    mode = "defect_implant"

    batch_size = 4
    dataloader = ds.create_dataloader(data_folder, csv_file, mode, batch_size,
     transforms=aug.generate_transforms_flip_affine(scales=(0.8, 1.2), degrees=(-45, 45)))
    print("Dataloader size: ", len(dataloader.dataset))

    for defects, implants, spacings in dataloader:
        print("Defects size: ", defects.size())
        print("Implants size: ", implants.size())
        print("Spacings size: ", spacings.size())

        defect, implant, spacing = defects[0, :, :, :], implants[0, :, :, :], spacings[0]

        print("Defect size: ", defect.size())
        print("Implant size: ", implant.size())
        print("Spacing: ", spacing)

        u.show_training_case(None, defect.numpy(), implant.numpy(), spacing.numpy())

def test_combined_dataloader_4():
    """
    Test whether the dataloader works for Combined training cases (with augmentation and iter size).
    """
    data_folder = p.combined_training_path
    csv_file = p.combined_training_csv_path
    mode = "defect_implant"

    batch_size = 4
    dataloader = ds.create_dataloader(data_folder, csv_file, mode, batch_size, iteration_size=100,
     transforms=aug.generate_transforms_flip_affine(scales=(0.8, 1.2), degrees=(-5, 5)))
    print("Dataloader size: ", len(dataloader.dataset))

    for defects, implants, spacings in dataloader:
        print("Defects size: ", defects.size())
        print("Implants size: ", implants.size())
        print("Spacings size: ", spacings.size())

        defect, implant, spacing = defects[0, :, :, :], implants[0, :, :, :], spacings[0]

        print("Defect size: ", defect.size())
        print("Implant size: ", implant.size())
        print("Spacing: ", spacing)

        u.show_training_case(None, defect.numpy(), implant.numpy(), spacing.numpy())

def run():
    # test_task_1_dataloader_1()
    # test_task_1_dataloader_2()
    # test_combined_dataloader_1()
    # test_combined_dataloader_2()
    # test_combined_dataloader_3()
    test_combined_dataloader_4()
    pass


if __name__ == "__main__":
    run()