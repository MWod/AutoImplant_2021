import os
import pathlib

import numpy as np
import pandas as pd

import paths as p
import utils as u

"""
Script performing the Task 1/2/3 dataset preprocessing (boundary cropping, resampling, padding to the same shape).
"""

def preprocess_task_1_training_set(input_data_folder : pathlib.Path, output_data_folder : pathlib.Path, csv_path : pathlib.Path, **preprocessing_params):
    output_spacing = preprocessing_params['output_spacing']
    output_size = preprocessing_params['output_size']
    pad_size = preprocessing_params['pad_size']
    offset = preprocessing_params['offset']

    dataframe = pd.read_csv(csv_path)
    print("Dataset size: ", len(dataframe))
    
    errors = list()
    for current_id, case in dataframe.iterrows():
        print("Current ID: ", current_id)
        complete_skull_path = input_data_folder / case['Complete Skull Path']
        defective_skull_path = input_data_folder / case['Defective Skull Path']
        implant_path = input_data_folder / case['Implant Path']

        complete_skull, defective_skull, implant, spacing = u.load_training_case(complete_skull_path, defective_skull_path, implant_path)
        print("Original Complete Skull Shape: ", complete_skull.shape)
        print("Original Defective Skull Shape: ", defective_skull.shape)
        print("Original Implant Shape: ", implant.shape)
        print("Initial spacing: ", spacing)

        preprocessed_complete_skull, preprocessed_defective_skull, preprocessed_implant, to_pad, internal_shape, padding = u.preprocess_training_case(defective_skull, complete_skull, implant, spacing, output_spacing, pad_size, output_size, offset)
        print("Preprocessed Complete Skull Shape: ", preprocessed_complete_skull.shape)
        print("Preprocessed Defective Skull Shape: ", preprocessed_defective_skull.shape)
        print("Preprocessed Implant Shape: ", preprocessed_implant.shape)

        recovered_complete_skull = u.postprocess_case(preprocessed_complete_skull, spacing, output_spacing, padding, to_pad, internal_shape, pad_size)
        mse = lambda a, b: np.mean((a-b)**2)
        error = mse(complete_skull, recovered_complete_skull)
        print("MSE: ", error)
        errors.append(error)

        preprocessed_complete_skull_path = output_data_folder / case['Complete Skull Path']
        preprocessed_defective_skull_path = output_data_folder / case['Defective Skull Path']
        preprocessed_implant_path = output_data_folder / case['Implant Path']

        pathlib.Path(preprocessed_complete_skull_path).parents[0].mkdir(parents=True, exist_ok=True)
        pathlib.Path(preprocessed_defective_skull_path).parents[0].mkdir(parents=True, exist_ok=True)
        pathlib.Path(preprocessed_implant_path).parents[0].mkdir(parents=True, exist_ok=True)

        u.save_volume(preprocessed_complete_skull, output_spacing, preprocessed_complete_skull_path)
        u.save_volume(preprocessed_defective_skull, output_spacing, preprocessed_defective_skull_path)
        u.save_volume(preprocessed_implant, output_spacing, preprocessed_implant_path)
    
    print("Mean error: ", np.mean(errors))
    print("Max error: ", np.max(errors))

def preprocess_task_1_testing_set(input_data_folder : pathlib.Path, output_data_folder : pathlib.Path, csv_path : pathlib.Path, **preprocessing_params):
    output_spacing = preprocessing_params['output_spacing']
    output_size = preprocessing_params['output_size']
    pad_size = preprocessing_params['pad_size']
    offset = preprocessing_params['offset']

    dataframe = pd.read_csv(csv_path)
    print("Dataset size: ", len(dataframe))
    
    errors = list()
    for current_id, case in dataframe.iterrows():
        print("Current ID: ", current_id)
        defective_skull_path = input_data_folder / case['Defective Skull Path']
        defective_skull, spacing = u.load_testing_case(defective_skull_path)
        print("Original Defective Skull Shape: ", defective_skull.shape)
        print("Initial spacing: ", spacing)

        preprocessed_defective_skull, to_pad, internal_shape, padding = u.preprocess_testing_case(defective_skull, spacing, output_spacing, pad_size, output_size, offset)
        print("Preprocessed Defective Skull Shape: ", preprocessed_defective_skull.shape)

        recovered_defective_skull = u.postprocess_case(preprocessed_defective_skull, spacing, output_spacing, padding, to_pad, internal_shape, pad_size)
        mse = lambda a, b: np.mean((a-b)**2)
        error = mse(defective_skull, recovered_defective_skull)
        print("MSE: ", error)
        errors.append(error)

        preprocessed_defective_skull_path = output_data_folder / case['Defective Skull Path']

        pathlib.Path(preprocessed_defective_skull_path).parents[0].mkdir(parents=True, exist_ok=True)

        u.save_volume(preprocessed_defective_skull, output_spacing, preprocessed_defective_skull_path)
    
    print("Mean error: ", np.mean(errors))
    print("Max error: ", np.max(errors))

def preprocess_task_2_testing_set(input_data_folder : pathlib.Path, output_data_folder : pathlib.Path, csv_path : pathlib.Path, **preprocessing_params):
    preprocess_task_1_testing_set(input_data_folder, output_data_folder, csv_path, **preprocessing_params)

def preprocess_task_3_training_set(input_data_folder : pathlib.Path, output_data_folder : pathlib.Path, csv_path : pathlib.Path, **preprocessing_params):
    preprocess_task_1_training_set(input_data_folder, output_data_folder, csv_path, **preprocessing_params)

def preprocess_task_3_testing_set(input_data_folder : pathlib.Path, output_data_folder : pathlib.Path, csv_path : pathlib.Path, **preprocessing_params):
    preprocess_task_1_testing_set(input_data_folder, output_data_folder, csv_path, **preprocessing_params)

def run():
    output_spacing = (1.0, 1.0, 1.0)
    output_size = (240, 200, 240)
    pad_size = 3
    offset = 35
    preprocessing_params = dict()
    preprocessing_params['output_spacing'] = output_spacing
    preprocessing_params['output_size'] = output_size
    preprocessing_params['pad_size'] = pad_size
    preprocessing_params['offset'] = offset

    preprocess_task_1_training_set(p.task_1_training_path, p.task_1_training_preprocessed_path, p.task_1_training_csv_path, **preprocessing_params)
    preprocess_task_1_training_set(p.task_1_training_path, p.task_1_training_preprocessed_path, p.task_1_validation_csv_path, **preprocessing_params)

    preprocess_task_3_training_set(p.task_3_training_path, p.task_3_training_preprocessed_path, p.task_3_training_csv_path, **preprocessing_params)
    preprocess_task_3_training_set(p.task_3_training_path, p.task_3_training_preprocessed_path, p.task_3_validation_csv_path, **preprocessing_params)


    output_spacing = (1.0, 1.0, 1.0)
    output_size = (240, 200, 240)
    pad_size = 3
    offset = 35
    preprocessing_params = dict()
    preprocessing_params['output_spacing'] = output_spacing
    preprocessing_params['output_size'] = output_size
    preprocessing_params['pad_size'] = pad_size
    preprocessing_params['offset'] = offset

    preprocess_task_1_testing_set(p.task_1_testing_path, p.task_1_testing_preprocessed_path, p.task_1_testing_csv_path, **preprocessing_params)
    preprocess_task_2_testing_set(p.task_2_testing_path, p.task_2_testing_preprocessed_path, p.task_2_testing_csv_path, **preprocessing_params)
    preprocess_task_3_testing_set(p.task_3_testing_path, p.task_3_testing_preprocessed_path, p.task_3_testing_csv_path, **preprocessing_params)


if __name__ == "__main__":
    run()