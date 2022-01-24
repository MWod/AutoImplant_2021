import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch as tc
import matplotlib.pyplot as plt

import utils as u
import paths as p
import evaluation_metrics as metrics

from networks import unet

import pipeline as pp


def test_pipeline_1():
    """
    Simple pipeline test (just defect reconstruction)
    """
    output_path = r"E:\Data\AutoImplant\Results\Simple_Tests\test_pipeline_1.nrrd"

    complete_path = r'E:\Data\AutoImplant\Task1\Training\complete_skull\bilateral\015.nrrd'
    defective_path = r'E:\Data\AutoImplant\Task1\Training\defective_skull\bilateral\015.nrrd'
    implant_path = r'E:\Data\AutoImplant\Task1\Training\implant\bilateral\015.nrrd'

    complete_skull, _, _ = u.load_volume(complete_path)
    defective_skull, _, _ = u.load_volume(defective_path)
    implant, spacing, _ = u.load_volume(implant_path)

    reconstruction_params = dict()
    reconstruction_params['device'] = "cuda:0"
    reconstruction_params['reconstruction_model'] = unet
    reconstruction_params['reconstruction_weights'] = p.combined_exp3_save_path / str("model_cp2")
    reconstruction_params['defect_refinement'] = False
    reconstruction_params['implant_modeling'] = False

    reconstructed_implant = pp.defect_reconstruction(defective_path, output_path, echo=True, **reconstruction_params)

    dc = metrics.dc(reconstructed_implant, implant)
    bdc = metrics.bdc(reconstructed_implant, implant, defective_skull, spacing)
    hd95 = metrics.hd95(reconstructed_implant, implant, spacing)

    print("Dice: ", np.mean(dc))
    print("Boundary Dice: ", np.mean(bdc))
    print("HD 95: ", np.mean(hd95))

    u.show_training_case(reconstructed_implant, defective_skull, implant, spacing, names=["Input", "Reconstruced Defect", "Real Defect"], show=False)
    u.show_training_case(complete_skull, defective_skull, np.logical_or(reconstructed_implant, defective_skull), spacing, names=["Input", "Complete Skull", "Reconstructed Skull"], show=False)
    plt.show()

def test_pipeline_2():
    """
    Simple pipeline test (just defect reconstruction)
    """
    output_path = None

    defective_path = r'E:\Data\AutoImplant\Separate_Test_Set\craniotomy skull\B0028\B0028.nrrd'

    defective_skull, spacing, _ = u.load_volume(defective_path)

    reconstruction_params = dict()
    reconstruction_params['device'] = "cuda:0"
    reconstruction_params['reconstruction_model'] = unet
    reconstruction_params['reconstruction_weights'] = p.combined_vae_exp1_save_path / str("model_cp7")
    reconstruction_params['defect_refinement'] = False
    reconstruction_params['implant_modeling'] = False
    reconstruction_params['initial_opening'] = True

    reconstructed_implant = pp.defect_reconstruction(defective_path, output_path, echo=True, **reconstruction_params)

    print("Spacing: ", spacing)

    u.show_training_case(reconstructed_implant, defective_skull, np.logical_or(defective_skull, reconstructed_implant), spacing, names=["Defective Skull", "Reconstructed Defect", "Complete Skull"], show=False)
    plt.show()


def run():
    # test_pipeline_1()
    test_pipeline_2()
    pass


if __name__ == "__main__":
    run()