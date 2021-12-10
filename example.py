import numpy as np
import matplotlib.pyplot as plt

import utils as u
import paths as p

from networks import unet

import pipeline as pp

def example():
    """
    Example presenting how to use the defect reconstruction / implant modeling pipeline.
    """
    output_path = None # TO DO - where to save the .nrrd file with the defect reconstruction / implant
    defective_path = None # TO DO - path to the defective skull

    reconstruction_params = dict()
    reconstruction_params['device'] = "cuda:0" # The GPU to use by the models, can be set to "cpu"
    reconstruction_params['reconstruction_model'] = unet # The model used for the defect reconstruction
    reconstruction_params['reconstruction_weights'] = None # Path to the pretrained model weights
    reconstruction_params['defect_refinement'] = False # Whether to use the defect refinement
    reconstruction_params['implant_modeling'] = False # Whether to use the implant modeling - Please note - task specific, will not work for other defects

    # reconstruction_params['refinement_model'] = unet # Refinement model if defect_refinement set to True
    # reconstruction_params['refinement_weights'] = Nne # Path to the pretrained refinement model if defect_refinement set to True

    reconstructed_implant = pp.defect_reconstruction(defective_path, output_path, echo=True, **reconstruction_params)

def run():
    example()

if __name__ == "__main__":
    run()