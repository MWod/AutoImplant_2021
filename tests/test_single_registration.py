import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import time

import numpy as np
import torch as tc
import matplotlib.pyplot as plt

import utils as u
import image_registration as reg


def test_registration_1():
    # source_path = r'E:\Data\AutoImplant\Task1\Training_Preprocessed\complete_skull\bilateral\024.nrrd'
    # target_path = r'E:\Data\AutoImplant\Task3\Training_Preprocessed\complete_skull\065.nrrd'
    # target_path = r'E:\Data\AutoImplant\Task1\Training_Preprocessed\complete_skull\random_1\024.nrrd' # offset problem
    # target_path = r'E:\Data\AutoImplant\Task1\Training_Preprocessed\complete_skull\random_1\029.nrrd'

    source_path = r'E:\Data\AutoImplant\Task1\Training_Preprocessed\complete_skull\bilateral\024.nrrd'
    target_path = r'E:\Data\AutoImplant\Task3\Training_Preprocessed\complete_skull\077.nrrd'

    source, spacing, _ = u.load_volume(source_path)
    target, _, _ = u.load_volume(target_path)

    b_t = time.time()
    displacement_field = reg.register_single(source, target)
    warped_source = u.image_warping(source, displacement_field)
    e_t = time.time()
    print("Time for registration: ", e_t - b_t, "seconds.")

    u.show_training_case(target, source, warped_source, spacing, names=["Source", "Target", "Warped Source"])

def run():
    test_registration_1()

if __name__ == "__main__":
    run()