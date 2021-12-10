import os
import sys
import pathlib
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
import time
import shutil

import numpy as np
import pandas as pd
import torch as tc

import paths as p
import utils as u

### Please Note - registration comes from external package that will be released separately ###
from registration import instance_optimization
from registration import utils_tc
from registration import cost_functions
from registration import regularizers

"""
Script performing the registration of Task 1/3 training sets.
"""

def register_single(source, target):
    """
    Registration (purely nonrigid) of the given source and target with fixed registration parameters.
    """
    device = "cuda:0"
    y_size, x_size, z_size = source.shape
    source = tc.from_numpy(source.astype(np.float32))
    target = tc.from_numpy(target.astype(np.float32))

    source = source.view(1, 1, y_size, x_size, z_size).to(device)
    target = target.view(1, 1, y_size, x_size, z_size).to(device)

    num_levels = 4
    used_levels = 4
    num_iters = 50
    learning_rate = 0.01
    alpha = 100000
    cost_function = cost_functions.mse_tc
    regularization_function = regularizers.diffusion_tc
    cost_function_params = {}
    regularization_function_params = {}
    
    transformation = instance_optimization.affine_registration(source, target, num_levels, used_levels-1, num_iters, learning_rate, cost_function, device=device)
    displacement_field_tc = utils_tc.tc_transform_to_tc_df(transformation, (1, 1, y_size, x_size, z_size), device=device)
    displacement_field_tc = instance_optimization.nonrigid_registration(source, target, num_levels, used_levels, num_iters, learning_rate,
    alpha, cost_function, regularization_function, cost_function_params, regularization_function_params, initial_displacement_field=displacement_field_tc, device=device)
    displacement_field = utils_tc.tc_df_to_np_df(displacement_field_tc)
    return displacement_field

def combine_datasets_by_registration():
    """
    Function to combine the Task 1 and Task 3 training sets using affine and nonrigid image registration.
    """
    task_1_dataframe = pd.read_csv(p.task_1_training_csv_path)
    task_3_dataframe = pd.read_csv(p.task_3_training_csv_path)

    print("Task 1 Size: ", len(task_1_dataframe))
    print("Task 3 Size: ", len(task_3_dataframe))

    task_1_data_path = p.task_1_training_preprocessed_path
    task_3_data_path = p.task_3_training_preprocessed_path
    combined_data_path = p.combined_training_path

    data = []
    # Task 1 vs Task 1
    all_pairs = list()
    all_defects = dict()
    for outer_id, outer_case in task_1_dataframe.iterrows():
        for inner_id, inner_case in task_1_dataframe.iterrows():
            inner_complete_path = inner_case['Complete Skull Path']
            outer_complete_path = outer_case['Complete Skull Path']

            inner_skull_id = os.path.split(inner_complete_path)[-1].split(".")[0]
            outer_skull_id = os.path.split(outer_complete_path)[-1].split(".")[0]

            if inner_skull_id != outer_skull_id:
                current_pair = (inner_skull_id, outer_skull_id)
                all_pairs.append((current_pair))
                try:
                    if inner_complete_path not in all_defects[current_pair]:
                        all_defects[current_pair].append(inner_complete_path)
                except:
                    all_defects[current_pair] = [inner_complete_path]

    all_pairs = list(set(all_pairs))
    print("Task 1/1 cases: ", len(all_pairs))

    i = 0
    for pair in all_pairs:
        print("Current pair: " + str(i) + " / " + str(len(all_pairs)))
        b_t = time.time()
        source_path = task_1_data_path / "complete_skull" / "bilateral" / pathlib.Path(pair[0]).with_suffix('.nrrd')
        target_path = task_1_data_path / "complete_skull" / "bilateral" / pathlib.Path(pair[1]).with_suffix('.nrrd')
        source, _, _ = u.load_volume(source_path)
        target, _, _ = u.load_volume(target_path)
        displacement_field = register_single(source, target)
        j = 0
        for to_warp_case in all_defects[pair]:
            complete_skull_path = task_1_data_path / to_warp_case
            defective_skull_path = task_1_data_path / to_warp_case.replace("complete_skull", "defective_skull")
            implant_path = task_1_data_path / to_warp_case.replace("complete_skull", "implant")
            complete_skull, spacing, _ = u.load_volume(complete_skull_path)
            defective_skull, _, _ = u.load_volume(defective_skull_path)
            implant, _, _ =  u.load_volume(implant_path)
            warped_complete_skull = u.image_warping(complete_skull, displacement_field, order=0)
            warped_defective_skull = u.image_warping(defective_skull, displacement_field, order=0)
            warped_implant = u.image_warping(implant, displacement_field, order=0)
            warped_complete_skull_path = pathlib.Path("complete_skull", "T11", str(i), str(j), "complete_skull.nrrd")
            warped_defective_skull_path = pathlib.Path("defective_skull", "T11", str(i), str(j), "defective_skull.nrrd")
            warped_implant_path = pathlib.Path("implant", "T11", str(i), str(j), "implant.nrrd")
            full_warped_complete_skull_path = combined_data_path / warped_complete_skull_path
            full_warped_defective_skull_path = combined_data_path / warped_defective_skull_path
            full_warped_implant_path = combined_data_path / warped_implant_path
            pathlib.Path(full_warped_complete_skull_path).parents[0].mkdir(parents=True, exist_ok=True)
            pathlib.Path(full_warped_defective_skull_path).parents[0].mkdir(parents=True, exist_ok=True)
            pathlib.Path(full_warped_implant_path).parents[0].mkdir(parents=True, exist_ok=True)
            u.save_volume(warped_complete_skull, spacing, full_warped_complete_skull_path)
            u.save_volume(warped_defective_skull, spacing, full_warped_defective_skull_path)
            u.save_volume(warped_implant , spacing, full_warped_implant_path)
            data.append([warped_complete_skull_path, warped_defective_skull_path, warped_implant_path])
            j += 1
        i += 1
        e_t = time.time()
        print("Time for current pair: ", e_t - b_t, " seconds.")
        print("Estimated time to end: ", (len(all_pairs) - i) * (e_t - b_t) / 60.0, " minutes.")

    # Task 3 vs Task 3 
    all_pairs = list()
    all_defects = dict()
    for outer_id, outer_case in task_3_dataframe.iterrows():
        for inner_id, inner_case in task_3_dataframe.iterrows():
            inner_complete_path = inner_case['Complete Skull Path']
            outer_complete_path = outer_case['Complete Skull Path']

            inner_skull_id = os.path.split(inner_complete_path)[-1].split(".")[0]
            outer_skull_id = os.path.split(outer_complete_path)[-1].split(".")[0]

            if inner_skull_id != outer_skull_id:
                current_pair = (inner_skull_id, outer_skull_id)
                all_pairs.append(current_pair)
                try:
                    if inner_complete_path not in all_defects[current_pair]:
                        all_defects[current_pair].append(inner_complete_path)
                except:
                    all_defects[current_pair] = [inner_complete_path]

    all_pairs = list(set(all_pairs))
    print("Task 3/3 cases: ", len(all_pairs))
    i = 0
    for pair in all_pairs:
        print("Current pair: " + str(i) + " / " + str(len(all_pairs)))
        b_t = time.time()
        source_path = task_3_data_path / "complete_skull" / pathlib.Path(pair[0]).with_suffix('.nrrd')
        target_path = task_3_data_path / "complete_skull" / pathlib.Path(pair[1]).with_suffix('.nrrd')
        source, _, _ = u.load_volume(source_path)
        target, _, _ = u.load_volume(target_path)
        displacement_field = register_single(source, target)
        j = 0
        for to_warp_case in all_defects[pair]:
            complete_skull_path = task_3_data_path / to_warp_case
            defective_skull_path = task_3_data_path / to_warp_case.replace("complete_skull", "defective_skull")
            implant_path = task_3_data_path / to_warp_case.replace("complete_skull", "implant")
            complete_skull, spacing, _ = u.load_volume(complete_skull_path)
            defective_skull, _, _ = u.load_volume(defective_skull_path)
            implant, _, _ =  u.load_volume(implant_path)
            warped_complete_skull = u.image_warping(complete_skull, displacement_field, order=0)
            warped_defective_skull = u.image_warping(defective_skull, displacement_field, order=0)
            warped_implant = u.image_warping(implant, displacement_field, order=0)
            warped_complete_skull_path = pathlib.Path("complete_skull", "T33", str(i), str(j), "complete_skull.nrrd")
            warped_defective_skull_path = pathlib.Path("defective_skull", "T33", str(i), str(j), "defective_skull.nrrd")
            warped_implant_path = pathlib.Path("implant", "T33", str(i), str(j), "implant.nrrd")
            full_warped_complete_skull_path = combined_data_path / warped_complete_skull_path
            full_warped_defective_skull_path = combined_data_path / warped_defective_skull_path
            full_warped_implant_path = combined_data_path / warped_implant_path
            pathlib.Path(full_warped_complete_skull_path).parents[0].mkdir(parents=True, exist_ok=True)
            pathlib.Path(full_warped_defective_skull_path).parents[0].mkdir(parents=True, exist_ok=True)
            pathlib.Path(full_warped_implant_path).parents[0].mkdir(parents=True, exist_ok=True)
            u.save_volume(warped_complete_skull, spacing, full_warped_complete_skull_path)
            u.save_volume(warped_defective_skull, spacing, full_warped_defective_skull_path)
            u.save_volume(warped_implant , spacing, full_warped_implant_path)
            data.append([warped_complete_skull_path, warped_defective_skull_path, warped_implant_path])
            j += 1
        i += 1
        e_t = time.time()
        print("Time for current pair: ", e_t - b_t, " seconds.")
        print("Estimated time to end: ", (len(all_pairs) - i) * (e_t - b_t) / 60.0, " minutes.")

    # Task 3 vs Task 1
    all_pairs = list()
    all_defects = dict()
    for outer_id, outer_case in task_1_dataframe.iterrows():
        for inner_id, inner_case in task_3_dataframe.iterrows():
            inner_complete_path = inner_case['Complete Skull Path']
            outer_complete_path = outer_case['Complete Skull Path']
            inner_skull_id = os.path.split(inner_complete_path)[-1].split(".")[0]
            outer_skull_id = os.path.split(outer_complete_path)[-1].split(".")[0]
            current_pair = (inner_skull_id, outer_skull_id)
            all_pairs.append(current_pair)
            try:
                if inner_complete_path not in all_defects[current_pair]:
                    all_defects[current_pair].append(inner_complete_path)
            except:
                all_defects[current_pair] = [inner_complete_path]

    all_pairs = list(set(all_pairs))
    print("Task 1/3 cases: ", len(all_pairs))
    i = 0
    for pair in all_pairs:
        print("Current pair: " + str(i) + " / " + str(len(all_pairs)))
        b_t = time.time()
        source_path = task_3_data_path / "complete_skull" / pathlib.Path(pair[0]).with_suffix('.nrrd')
        target_path = task_1_data_path / "complete_skull" / "bilateral" / pathlib.Path(pair[1]).with_suffix('.nrrd')
        source, _, _ = u.load_volume(source_path)
        target, _, _ = u.load_volume(target_path)
        displacement_field = register_single(source, target)
        j = 0
        for to_warp_case in all_defects[pair]:
            complete_skull_path = task_3_data_path / to_warp_case
            defective_skull_path = task_3_data_path / to_warp_case.replace("complete_skull", "defective_skull")
            implant_path = task_3_data_path / to_warp_case.replace("complete_skull", "implant")
            complete_skull, spacing, _ = u.load_volume(complete_skull_path)
            defective_skull, _, _ = u.load_volume(defective_skull_path)
            implant, _, _ =  u.load_volume(implant_path)
            warped_complete_skull = u.image_warping(complete_skull, displacement_field, order=0)
            warped_defective_skull = u.image_warping(defective_skull, displacement_field, order=0)
            warped_implant = u.image_warping(implant, displacement_field, order=0)
            warped_complete_skull_path = pathlib.Path("complete_skull", "T31", str(i), str(j), "complete_skull.nrrd")
            warped_defective_skull_path = pathlib.Path("defective_skull", "T31", str(i), str(j), "defective_skull.nrrd")
            warped_implant_path = pathlib.Path("implant", "T31", str(i), str(j), "implant.nrrd")
            full_warped_complete_skull_path = combined_data_path / warped_complete_skull_path
            full_warped_defective_skull_path = combined_data_path / warped_defective_skull_path
            full_warped_implant_path = combined_data_path / warped_implant_path
            pathlib.Path(full_warped_complete_skull_path).parents[0].mkdir(parents=True, exist_ok=True)
            pathlib.Path(full_warped_defective_skull_path).parents[0].mkdir(parents=True, exist_ok=True)
            pathlib.Path(full_warped_implant_path).parents[0].mkdir(parents=True, exist_ok=True)
            u.save_volume(warped_complete_skull, spacing, full_warped_complete_skull_path)
            u.save_volume(warped_defective_skull, spacing, full_warped_defective_skull_path)
            u.save_volume(warped_implant , spacing, full_warped_implant_path)
            data.append([warped_complete_skull_path, warped_defective_skull_path, warped_implant_path])
            j += 1
        i += 1
        e_t = time.time()
        print("Time for current pair: ", e_t - b_t, " seconds.")
        print("Estimated time to end: ", (len(all_pairs) - i) * (e_t - b_t) / 60.0, " minutes.")

    # Task 1 vs Task 3
    all_pairs = list()
    all_defects = dict()
    for outer_id, outer_case in task_3_dataframe.iterrows():
        for inner_id, inner_case in task_1_dataframe.iterrows():
            inner_complete_path = inner_case['Complete Skull Path']
            outer_complete_path = outer_case['Complete Skull Path']
            inner_skull_id = os.path.split(inner_complete_path)[-1].split(".")[0]
            outer_skull_id = os.path.split(outer_complete_path)[-1].split(".")[0]
            current_pair = (inner_skull_id, outer_skull_id)
            all_pairs.append(current_pair)
            try:
                if inner_complete_path not in all_defects[current_pair]:
                    all_defects[current_pair].append(inner_complete_path)
            except:
                all_defects[current_pair] = [inner_complete_path]

    all_pairs = list(set(all_pairs))
    print("Task 3/1 cases: ",len(all_pairs))
    i = 0
    for pair in all_pairs:
        print("Current pair: " + str(i) + " / " + str(len(all_pairs)))
        b_t = time.time()
        source_path = task_1_data_path / "complete_skull" / "bilateral" / pathlib.Path(pair[0]).with_suffix('.nrrd')
        target_path = task_3_data_path / "complete_skull" / pathlib.Path(pair[1]).with_suffix('.nrrd')
        source, _, _ = u.load_volume(source_path)
        target, _, _ = u.load_volume(target_path)
        displacement_field = register_single(source, target)
        j = 0
        for to_warp_case in all_defects[pair]:
            complete_skull_path = task_1_data_path / to_warp_case
            defective_skull_path = task_1_data_path / to_warp_case.replace("complete_skull", "defective_skull")
            implant_path = task_1_data_path / to_warp_case.replace("complete_skull", "implant")
            complete_skull, spacing, _ = u.load_volume(complete_skull_path)
            defective_skull, _, _ = u.load_volume(defective_skull_path)
            implant, _, _ =  u.load_volume(implant_path)
            warped_complete_skull = u.image_warping(complete_skull, displacement_field, order=0)
            warped_defective_skull = u.image_warping(defective_skull, displacement_field, order=0)
            warped_implant = u.image_warping(implant, displacement_field, order=0)
            warped_complete_skull_path = pathlib.Path("complete_skull", "T13", str(i), str(j), "complete_skull.nrrd")
            warped_defective_skull_path = pathlib.Path("defective_skull", "T13", str(i), str(j), "defective_skull.nrrd")
            warped_implant_path = pathlib.Path("implant", "T13", str(i), str(j), "implant.nrrd")
            full_warped_complete_skull_path = combined_data_path / warped_complete_skull_path
            full_warped_defective_skull_path = combined_data_path / warped_defective_skull_path
            full_warped_implant_path = combined_data_path / warped_implant_path
            pathlib.Path(full_warped_complete_skull_path).parents[0].mkdir(parents=True, exist_ok=True)
            pathlib.Path(full_warped_defective_skull_path).parents[0].mkdir(parents=True, exist_ok=True)
            pathlib.Path(full_warped_implant_path).parents[0].mkdir(parents=True, exist_ok=True)
            u.save_volume(warped_complete_skull, spacing, full_warped_complete_skull_path)
            u.save_volume(warped_defective_skull, spacing, full_warped_defective_skull_path)
            u.save_volume(warped_implant , spacing, full_warped_implant_path)
            data.append([warped_complete_skull_path, warped_defective_skull_path, warped_implant_path])
            j += 1
        i += 1
        e_t = time.time()
        print("Time for current pair: ", e_t - b_t, " seconds.")
        print("Estimated time to end: ", (len(all_pairs) - i) * (e_t - b_t) / 60.0, " minutes.")

    output_dataframe = pd.DataFrame(data, columns=['Complete Skull Path', "Defective Skull Path", "Implant Path"])
    output_dataframe.to_csv(p.combined_training_csv_path, index=False)


def combine_validation_sets():
    """
    Function to combine Task 1/3 validation sets (w/o registration).
    """
    task_1_val_dataframe = pd.read_csv(p.task_1_validation_csv_path)
    task_3_val_dataframe = pd.read_csv(p.task_3_validation_csv_path)

    task_1_data_path = p.task_1_training_preprocessed_path
    task_3_data_path = p.task_3_training_preprocessed_path
    combined_data_path = p.combined_training_path

    data = []
    for current_id, case in task_1_val_dataframe.iterrows():
        input_complete_skull_path = task_1_data_path / case['Complete Skull Path']
        input_defective_skull_path = task_1_data_path / case['Defective Skull Path']
        input_implant_path = task_1_data_path / case['Implant Path']

        output_complete_skull_path = combined_data_path / "Validation" / case['Complete Skull Path']
        output_defective_skull_path = combined_data_path / "Validation" / case['Defective Skull Path']
        output_implant_path = combined_data_path / "Validation" / case['Implant Path']

        pathlib.Path(output_complete_skull_path).parents[0].mkdir(parents=True, exist_ok=True)
        pathlib.Path(output_defective_skull_path).parents[0].mkdir(parents=True, exist_ok=True)
        pathlib.Path(output_implant_path).parents[0].mkdir(parents=True, exist_ok=True)

        shutil.copy(input_complete_skull_path, output_complete_skull_path)
        shutil.copy(input_defective_skull_path, output_defective_skull_path)
        shutil.copy(input_implant_path, output_implant_path)

        data.append([output_complete_skull_path, output_defective_skull_path, output_implant_path])

    for current_id, case in task_3_val_dataframe.iterrows():
        input_complete_skull_path = task_3_data_path / case['Complete Skull Path']
        input_defective_skull_path = task_3_data_path / case['Defective Skull Path']
        input_implant_path = task_3_data_path / case['Implant Path']

        output_complete_skull_path = combined_data_path / "Validation" / case['Complete Skull Path']
        output_defective_skull_path = combined_data_path / "Validation" / case['Defective Skull Path']
        output_implant_path = combined_data_path / "Validation" / case['Implant Path']

        pathlib.Path(output_complete_skull_path).parents[0].mkdir(parents=True, exist_ok=True)
        pathlib.Path(output_defective_skull_path).parents[0].mkdir(parents=True, exist_ok=True)
        pathlib.Path(output_implant_path).parents[0].mkdir(parents=True, exist_ok=True)

        shutil.copy(input_complete_skull_path, output_complete_skull_path)
        shutil.copy(input_defective_skull_path, output_defective_skull_path)
        shutil.copy(input_implant_path, output_implant_path)

        data.append([output_complete_skull_path, output_defective_skull_path, output_implant_path])
    
    validation_dataframe = pd.DataFrame(data, columns=['Complete Skull Path', "Defective Skull Path", "Implant Path"])
    validation_dataframe.to_csv(p.combined_validation_csv_path, index=False)

def combine_training_sets_wor():
    """
    Function to combine Task 1/3 training sets (w/o registration).
    """
    task_1_tr_dataframe = pd.read_csv(p.task_1_training_csv_path)
    task_3_tr_dataframe = pd.read_csv(p.task_3_training_csv_path)

    task_1_data_path = p.task_1_training_preprocessed_path
    task_3_data_path = p.task_3_training_preprocessed_path
    combined_data_path = p.combined_training_path

    data = []
    for current_id, case in task_1_tr_dataframe.iterrows():
        input_complete_skull_path = task_1_data_path / case['Complete Skull Path']
        input_defective_skull_path = task_1_data_path / case['Defective Skull Path']
        input_implant_path = task_1_data_path / case['Implant Path']

        output_complete_skull_path = combined_data_path / "TRWR" / case['Complete Skull Path']
        output_defective_skull_path = combined_data_path / "TRWR" / case['Defective Skull Path']
        output_implant_path = combined_data_path / "TRWR" / case['Implant Path']

        pathlib.Path(output_complete_skull_path).parents[0].mkdir(parents=True, exist_ok=True)
        pathlib.Path(output_defective_skull_path).parents[0].mkdir(parents=True, exist_ok=True)
        pathlib.Path(output_implant_path).parents[0].mkdir(parents=True, exist_ok=True)

        shutil.copy(input_complete_skull_path, output_complete_skull_path)
        shutil.copy(input_defective_skull_path, output_defective_skull_path)
        shutil.copy(input_implant_path, output_implant_path)

        data.append([output_complete_skull_path, output_defective_skull_path, output_implant_path])

    for current_id, case in task_3_tr_dataframe.iterrows():
        input_complete_skull_path = task_3_data_path / case['Complete Skull Path']
        input_defective_skull_path = task_3_data_path / case['Defective Skull Path']
        input_implant_path = task_3_data_path / case['Implant Path']

        output_complete_skull_path = combined_data_path / "TRWR" / case['Complete Skull Path']
        output_defective_skull_path = combined_data_path / "TRWR" / case['Defective Skull Path']
        output_implant_path = combined_data_path / "TRWR" / case['Implant Path']

        pathlib.Path(output_complete_skull_path).parents[0].mkdir(parents=True, exist_ok=True)
        pathlib.Path(output_defective_skull_path).parents[0].mkdir(parents=True, exist_ok=True)
        pathlib.Path(output_implant_path).parents[0].mkdir(parents=True, exist_ok=True)

        shutil.copy(input_complete_skull_path, output_complete_skull_path)
        shutil.copy(input_defective_skull_path, output_defective_skull_path)
        shutil.copy(input_implant_path, output_implant_path)

        data.append([output_complete_skull_path, output_defective_skull_path, output_implant_path])
    
    training_dataframe = pd.DataFrame(data, columns=['Complete Skull Path', "Defective Skull Path", "Implant Path"])
    training_dataframe.to_csv(p.combined_training_wr_csv_path, index=False)

def combine_validation_sets():
    """
    Function to combine Task 1/3 validation sets (w/o registration).
    """
    task_1_tr_dataframe = pd.read_csv(p.task_1_validation_csv_path)
    task_3_tr_dataframe = pd.read_csv(p.task_3_validation_csv_path)

    task_1_data_path = p.task_1_training_preprocessed_path
    task_3_data_path = p.task_3_training_preprocessed_path
    combined_data_path = p.combined_training_path

    data = []
    for current_id, case in task_1_tr_dataframe.iterrows():
        input_complete_skull_path = task_1_data_path / case['Complete Skull Path']
        input_defective_skull_path = task_1_data_path / case['Defective Skull Path']
        input_implant_path = task_1_data_path / case['Implant Path']

        output_complete_skull_path = combined_data_path / "Validation" / case['Complete Skull Path']
        output_defective_skull_path = combined_data_path / "Validation" / case['Defective Skull Path']
        output_implant_path = combined_data_path / "Validation" / case['Implant Path']

        pathlib.Path(output_complete_skull_path).parents[0].mkdir(parents=True, exist_ok=True)
        pathlib.Path(output_defective_skull_path).parents[0].mkdir(parents=True, exist_ok=True)
        pathlib.Path(output_implant_path).parents[0].mkdir(parents=True, exist_ok=True)

        shutil.copy(input_complete_skull_path, output_complete_skull_path)
        shutil.copy(input_defective_skull_path, output_defective_skull_path)
        shutil.copy(input_implant_path, output_implant_path)

        data.append([output_complete_skull_path, output_defective_skull_path, output_implant_path])

    for current_id, case in task_3_tr_dataframe.iterrows():
        input_complete_skull_path = task_3_data_path / case['Complete Skull Path']
        input_defective_skull_path = task_3_data_path / case['Defective Skull Path']
        input_implant_path = task_3_data_path / case['Implant Path']

        output_complete_skull_path = combined_data_path / "Validation" / case['Complete Skull Path']
        output_defective_skull_path = combined_data_path / "Validation" / case['Defective Skull Path']
        output_implant_path = combined_data_path / "Validation" / case['Implant Path']

        pathlib.Path(output_complete_skull_path).parents[0].mkdir(parents=True, exist_ok=True)
        pathlib.Path(output_defective_skull_path).parents[0].mkdir(parents=True, exist_ok=True)
        pathlib.Path(output_implant_path).parents[0].mkdir(parents=True, exist_ok=True)

        shutil.copy(input_complete_skull_path, output_complete_skull_path)
        shutil.copy(input_defective_skull_path, output_defective_skull_path)
        shutil.copy(input_implant_path, output_implant_path)

        data.append([output_complete_skull_path, output_defective_skull_path, output_implant_path])
    
    validation_dataframe = pd.DataFrame(data, columns=['Complete Skull Path', "Defective Skull Path", "Implant Path"])
    validation_dataframe.to_csv(p.combined_validation_csv_path, index=False)

def run():
    # combine_datasets_by_registration()
    # combine_training_sets_wor()
    # combine_validation_sets()
    pass

if __name__ == "__main__":
    run()