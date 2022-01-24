import os
import pathlib

import numpy as np
import scipy.ndimage as nd
import torch as tc

import reconstruction_to_implant as rti
import utils as u


def defect_reconstruction(defective_skull_path, reconstructed_implant_path, echo=False, **reconstruction_params):
    ### General params load and check
    print("Params loading and checking..") if echo else None
    device = reconstruction_params['device']

    ### Preprocessing params 
    try:
        reconstruction_spacing = reconstruction_params['reconstruction_spacing']
    except KeyError:
        reconstruction_spacing = (1.0, 1.0, 1.0)
    try:
        reconstruction_size = reconstruction_params['reconstruction_size']
    except KeyError:
        reconstruction_size = (240, 200, 240)
    try:
        reconstruction_pad_size = reconstruction_params['reconstruction_pad_size']
    except KeyError:
        reconstruction_pad_size = 3
    try:
        reconstruction_offset = reconstruction_params['reconstruction_offset']
    except KeyError:
        reconstruction_offset = 3
    try:
        initial_opening = reconstruction_params['initial_opening']
    except KeyError:
        initial_opening = False

    ### Defect reconstruction params and model creation
    reconstruction_model = reconstruction_params['reconstruction_model']
    reconstruction_model_weights = reconstruction_params['reconstruction_weights']
    model = reconstruction_model.load_network().to(device)
    state_dict = u.parse_state_dict(str(reconstruction_model_weights))
    model.load_state_dict(state_dict)
    model.eval()

    ### Defect refinement params (optional)
    defect_refinement = reconstruction_params['defect_refinement']
    if defect_refinement:
        try:
            output_shape = reconstruction_params['refinement_output_shape']
        except KeyError:
            output_shape = (200, 200, 200)
        try:
            boundary_offset = reconstruction_params['refinement_boundary_offset']
        except KeyError:
            boundary_offset = 10
        try:
            postprocess_refinement = reconstruction_params['postprocess_refinement']
        except KeyError:
            postprocess_refinement = True
        refinement_network_model = reconstruction_params['refinement_model']
        refinement_model_weights = reconstruction_params['refinement_weights']
        refinement_model = refinement_network_model.load_network().to(device)
        refinement_state_dict = u.parse_state_dict(str(refinement_model_weights))
        refinement_model.load_state_dict(refinement_state_dict)
        refinement_model.eval()

    ### Implant reconsstruction params (optional)
    implant_modeling = reconstruction_params['implant_modeling']
    if implant_modeling:
        try:
            desired_ratio = reconstruction_params['desired_ratio']
        except KeyError:
            desired_ratio = 0.7

    ### Load the data
    print("Data loading..") if echo else None
    defective_skull, spacing, metadata = u.load_volume(defective_skull_path, load_origin=True, load_direction=True)

    ### Preprocess the defective skull
    print("Data preprocessing..") if echo else None
    preprocessed_defective_skull, to_pad, internal_shape, padding = u.preprocess_testing_case(defective_skull, spacing,
        reconstruction_spacing, reconstruction_pad_size, reconstruction_size, reconstruction_offset)

    if initial_opening:
        preprocessed_defective_skull = nd.binary_opening(preprocessed_defective_skull)

    ### Perform the defect reconstruction
    print("Defect reconstruction..") if echo else None
    reconstruction_input = tc.from_numpy(preprocessed_defective_skull.astype(np.float32)).view(1, 1, *preprocessed_defective_skull.shape).to(device)
    with tc.set_grad_enabled(False):
        implant = model(reconstruction_input)
    implant = implant[0, 0, :, :, :].detach().cpu().numpy() > 0.5

    ### Resample to original resolution and postprocess
    print("Defect postprocessing..") if echo else None
    implant = u.postprocess_case(implant, spacing, reconstruction_spacing, padding, to_pad, internal_shape, reconstruction_pad_size)
    implant = u.binary_postprocessing(implant, defective_skull)

    ### Perform the defect refinement (optional)
    if defect_refinement:
        print("Defect refinement..") if echo else None
        implant = u.defect_refinement(implant, refinement_model, output_shape, boundary_offset, device=device)
        if postprocess_refinement:
            print("Refinement postprocessing..") if echo else None
            implant = u.binary_postprocessing(implant, defective_skull)
        
    ### Perform the implant modeling (optional)
    if implant_modeling:
        print("Implant refinement..") if echo else None
        implant = rti.reconstruction_to_implant(implant, defective_skull, desired_ratio=desired_ratio)

    ### Save the output
    if reconstructed_implant_path is not None:
        print("Data saving..") if echo else None
        pathlib.Path(reconstructed_implant_path).parents[0].mkdir(parents=True, exist_ok=True)
        u.save_volume(implant, spacing, reconstructed_implant_path, origin=metadata['origin'], direction=metadata['direction'])

    print("Processing finished.") if echo else None
    return implant

def run():
    pass

if __name__ == "__main__":
    run()