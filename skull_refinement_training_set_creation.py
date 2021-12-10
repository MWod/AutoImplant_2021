import os
import pathlib

import numpy as np
import matplotlib.pyplot as plt
import torch as tc
import pandas as pd

import paths as p
import utils as u

from networks import unet

import pipeline as pp

def crop_and_resample(reconstructed_defect, complete_skull, implant, defective_skull, input_spacing, output_shape, boundary_offset, mode='defect_complete', device="cpu"):
    boundaries = u.find_boundaries(reconstructed_defect, offset=boundary_offset)
    b_y, e_y, b_x, e_x, b_z, e_z = boundaries
    if mode == 'defect_complete':
        processed_defective_skull = np.logical_or(reconstructed_defect, defective_skull)[b_y:e_y, b_x:e_x, b_z:e_z]
    else:
        processed_defective_skull = reconstructed_defect[b_y:e_y, b_x:e_x, b_z:e_z]
    processed_complete_skull = complete_skull[b_y:e_y, b_x:e_x, b_z:e_z]
    processed_implant = implant[b_y:e_y, b_x:e_x, b_z:e_z]
    original_shape = processed_defective_skull.shape
    processed_defective_skull = u.resample_to_shape(processed_defective_skull, output_shape)
    processed_complete_skull = u.resample_to_shape(processed_complete_skull, output_shape)
    processed_implant = u.resample_to_shape(processed_implant, output_shape)
    output_spacing = input_spacing
    return processed_defective_skull, processed_complete_skull, processed_implant, output_spacing, original_shape

def prepare_dataset_for_second_stage(input_data_folders, output_data_folder, csv_paths, output_csv_path, model_files, weights_paths, subset_names,
    output_shape, boundary_offset, mode='defect_complete', dtype=tc.float32, device="cpu"):

    num_models = len(model_files)
    data = list()
    for i in range(num_models):
        dataframe = pd.read_csv(csv_paths[i])
        
        reconstruction_params = dict()
        reconstruction_params['device'] = device
        reconstruction_params['reconstruction_model'] = model_files[i]
        reconstruction_params['reconstruction_weights'] = weights_paths[i]
        reconstruction_params['defect_refinement'] = False
        reconstruction_params['implant_modeling'] = False

        subset_name = subset_names[i]
        for _, case in dataframe.iterrows():
            complete_path = input_data_folders[i] / case['Complete Skull Path']
            defective_path = input_data_folders[i] / case['Defective Skull Path']
            implant_path = input_data_folders[i] / case['Implant Path']

            complete_skull, _, _ = u.load_volume(complete_path)
            defective_skull, _, _ = u.load_volume(defective_path)
            implant, spacing, _ = u.load_volume(implant_path)

            reconstructed_implant = pp.defect_reconstruction(defective_path, None, echo=True, **reconstruction_params)

            processed_defective_skull, processed_complete_skull, processed_implant, output_spacing, original_shape = crop_and_resample(reconstructed_implant,
                complete_skull, implant, defective_skull, spacing, output_shape, boundary_offset, mode=mode, device=device)

            # u.show_training_case(processed_complete_skull, processed_defective_skull, processed_implant, output_spacing, show=True)

            processed_complete_skull_path = output_data_folder / subset_name / case['Complete Skull Path']
            processed_defective_skull_path = output_data_folder / subset_name / case['Defective Skull Path']
            processed_implant_path = output_data_folder / subset_name / case['Implant Path']

            pathlib.Path(processed_complete_skull_path).parents[0].mkdir(parents=True, exist_ok=True)
            pathlib.Path(processed_defective_skull_path).parents[0].mkdir(parents=True, exist_ok=True)
            pathlib.Path(processed_implant_path).parents[0].mkdir(parents=True, exist_ok=True)

            u.save_volume(processed_complete_skull, output_spacing, processed_complete_skull_path)
            u.save_volume(processed_defective_skull, output_spacing, processed_defective_skull_path)
            u.save_volume(processed_implant, output_spacing, processed_implant_path)

            data.append([os.path.join(subset_name, case['Complete Skull Path']), os.path.join(subset_name, case['Defective Skull Path']), os.path.join(subset_name, case['Implant Path'])])
    dataframe = pd.DataFrame(data, columns=['Complete Skull Path', "Defective Skull Path", "Implant Path"])
    dataframe.to_csv(output_csv_path, index=False)


def run():
    output_shape = (200, 200, 200)
    boundary_offset = 10
    network_model = unet
    device = "cuda:0"

    output_data_folder = p.second_step_implant_training_path
    mode = 'defect_implant'

    subset_names = ["Task1_Validation", "Task3_Validation"]
    input_data_folders = [p.task_1_training_path, p.task_3_training_path]
    input_csv_paths = [p.task_1_validation_csv_path, p.task_3_validation_csv_path]
    output_csv_path = p.second_step_validation_csv_path
    model_files = [network_model, network_model]
    weights_paths = [p.combined_exp3_save_path / str("model_cp2"), p.combined_exp3_save_path / str("model_cp2")]

    prepare_dataset_for_second_stage(input_data_folders, output_data_folder, input_csv_paths, output_csv_path, model_files, weights_paths,
     subset_names, output_shape, boundary_offset, mode=mode, device=device)
     
    subset_names = ["Task1_Training", "Task3_Training"]
    input_data_folders = [p.task_1_training_path, p.task_3_training_path]
    input_csv_paths = [p.task_1_training_csv_path, p.task_3_training_csv_path]
    output_csv_path = p.second_step_training_csv_path
    model_files = [network_model, network_model]
    weights_paths = [p.combined_exp3_save_path / str("model_cp2"), p.combined_exp3_save_path / str("model_cp2")]

    prepare_dataset_for_second_stage(input_data_folders, output_data_folder, input_csv_paths, output_csv_path, model_files, weights_paths,
     subset_names, output_shape, boundary_offset, mode=mode, device=device)




if __name__ == "__main__":
    run()
