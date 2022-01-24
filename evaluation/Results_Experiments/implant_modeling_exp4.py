"""
Experiment description:

Implant Modeling - Exp 3 (with refinement)
"""

import os
import sys
import getopt
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import utils as u
import paths as p
from networks import unet

import run_evaluation as re

def evaluation():
    echo = True
    output_data_path = p.implant_modeling_exp4_results_path

    reconstruction_params = dict()
    reconstruction_params['device'] = "cuda:0"
    reconstruction_params['reconstruction_model'] = unet
    reconstruction_params['reconstruction_weights'] = p.combined_vae_exp1_save_path / str("model_cp7")
    reconstruction_params['defect_refinement'] = True
    reconstruction_params['implant_modeling'] = True

    reconstruction_params['refinement_model'] = unet
    reconstruction_params['refinement_weights'] = p.second_step_exp3_save_path / str("model_cp2")

    reconstruction_params['desired_ratio'] = 0.7

    re.run_evaluation_on_testing_set(output_data_path=output_data_path, reconstruction_params=reconstruction_params, echo=echo)


if __name__ == "__main__":
    evaluation()