import os
import pathlib

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import paths as p
import utils as u

"""
Scripts used to create .csv files from the original dataset structure, as well as after the preprocessing, registration, and augmentation
"""

def create_task_1_training_csv(data_folder : pathlib.Path, output_csv_path : pathlib.Path):
    complete_skull_path = pathlib.PurePath(data_folder, "complete_skull")
    defective_skull_path = pathlib.PurePath(data_folder, "defective_skull")

    cases = os.listdir(complete_skull_path / "bilateral")
    cases = [item for item in cases if ".nrrd" in item]
    defect_types = os.listdir(defective_skull_path)

    data = list()
    for defect_type in defect_types:
        for case in cases:
            complete_skull_path = pathlib.Path("complete_skull", defect_type, case)
            defective_skull_path = pathlib.Path("defective_skull", defect_type, case)
            implant_path = pathlib.Path("implant", defect_type, case)
            data.append([complete_skull_path, defective_skull_path, implant_path])
    dataframe = pd.DataFrame(data, columns=['Complete Skull Path', "Defective Skull Path", "Implant Path"])
    dataframe.to_csv(output_csv_path, index=False)

def create_task_1_testing_csv(data_folder : pathlib.Path, output_csv_path : pathlib.Path):
    defective_skull_path = pathlib.PurePath(data_folder, "defective_skull")
    defect_types = os.listdir(defective_skull_path)

    data = list()
    for defect_type in defect_types:
        cases = os.listdir(defective_skull_path / defect_type)
        cases = [item for item in cases if ".nrrd" in item]
        for case in cases:
            skull_path = pathlib.Path("defective_skull", defect_type, case)
            data.append([skull_path])
    dataframe = pd.DataFrame(data, columns=["Defective Skull Path"])
    dataframe.to_csv(output_csv_path, index=False)

def create_task_2_testing_csv(data_folder : pathlib.Path, output_csv_path : pathlib.Path):
    cases = os.listdir(data_folder)
    cases = [item for item in cases if ".nrrd" in item]
    data = list()
    for case in cases:
        skull_path = pathlib.Path(case)
        data.append([skull_path])
    dataframe = pd.DataFrame(data, columns=["Defective Skull Path"])
    dataframe.to_csv(output_csv_path, index=False)

def create_task_3_training_csv(data_folder : pathlib.Path, output_csv_path : pathlib.Path):
    complete_skull_path = pathlib.PurePath(data_folder, "complete_skull")
    cases = os.listdir(complete_skull_path)
    cases = [item for item in cases if ".nrrd" in item]
    data = list()
    for case in cases:
        complete_skull_path = pathlib.Path("complete_skull", case)
        defective_skull_path = pathlib.Path("defective_skull", case)
        implant_path = pathlib.Path("implant", case)
        data.append([complete_skull_path, defective_skull_path, implant_path])
    dataframe = pd.DataFrame(data, columns=['Complete Skull Path', "Defective Skull Path", "Implant Path"])
    dataframe.to_csv(output_csv_path, index=False)

def create_task_3_testing_csv(data_folder : pathlib.Path, output_csv_path : pathlib.Path):
    cases = os.listdir(data_folder)
    cases = [item for item in cases if ".nrrd" in item]
    data = list()
    for case in cases:
        skull_path = pathlib.Path(case)
        data.append([skull_path])
    dataframe = pd.DataFrame(data, columns=["Defective Skull Path"])
    dataframe.to_csv(output_csv_path, index=False)

def split_training_validation(input_csv_path : pathlib.Path, output_training_csv_path : pathlib.Path, output_validation_csv_path : pathlib.Path, split_ratio : float):
    dataframe = pd.read_csv(input_csv_path)
    np.random.seed(12345)
    training_indices = np.random.rand(len(dataframe)) < split_ratio
    validation_indices = np.logical_not(training_indices)
    training_dataframe = dataframe[training_indices]
    validation_dataframe = dataframe[validation_indices]
    training_dataframe.to_csv(output_training_csv_path, index=False)
    validation_dataframe.to_csv(output_validation_csv_path, index=False)

def training_set_summary(data_folder : pathlib.Path, csv_path : pathlib.Path, dataset_name : str = "", show : bool=False):
    dataframe = pd.read_csv(csv_path)

    print("Summary of dataset:", dataset_name)
    print("Dataset size: ", len(dataframe))

    for current_id, case in dataframe.iterrows():
        complete_skull_path = data_folder / case['Complete Skull Path']
        defective_skull_path = data_folder / case['Defective Skull Path']
        implant_path = data_folder / case['Implant Path']

        complete_skull, defective_skull, implant, spacing = u.load_training_case(complete_skull_path, defective_skull_path, implant_path)
        print("Current ID: ", current_id, "Shape: ", complete_skull.shape, "Spacing: ", spacing)
        if defective_skull.shape != complete_skull.shape or defective_skull.shape != implant.shape:
            raise ValueError("Images do not have the same shape.")

        if show:
            u.show_training_case(complete_skull, defective_skull, implant, spacing)
            plt.close()

def testing_set_summary(data_folder : pathlib.Path, csv_path : pathlib.Path, dataset_name : str = "", show : bool=False):
    dataframe = pd.read_csv(csv_path)

    print("Summary of dataset:", dataset_name)
    print("Dataset size: ", len(dataframe))

    for current_id, case in dataframe.iterrows():
        defective_skull_path = data_folder / case['Defective Skull Path']

        defective_skull, spacing = u.load_testing_case(defective_skull_path)
        print("Current ID: ", current_id, "Shape: ", defective_skull.shape, "Spacing: ", spacing)
        if show:
            u.show_training_case(None, defective_skull, None, spacing)
            plt.close()


def run():
    # create_task_1_training_csv(p.task_1_training_path, p.task_1_dataset_csv_path)
    # split_training_validation(p.task_1_dataset_csv_path, p.task_1_training_csv_path, p.task_1_validation_csv_path, 0.9)
    # create_task_1_testing_csv(p.task_1_testing_path, p.task_1_testing_csv_path)

    # create_task_2_testing_csv(p.task_2_testing_path, p.task_2_testing_csv_path)

    # create_task_3_training_csv(p.task_3_training_path, p.task_3_dataset_csv_path)
    # split_training_validation(p.task_3_dataset_csv_path, p.task_3_training_csv_path, p.task_3_validation_csv_path, 0.9)
    # create_task_3_testing_csv(p.task_3_testing_path, p.task_3_testing_csv_path)

    # training_set_summary(p.task_1_training_path, p.task_1_training_csv_path, "Task 1 Training Set")
    # training_set_summary(p.task_1_training_path, p.task_1_validation_csv_path, "Task 1 Validation Set")
    # training_set_summary(p.task_3_training_path, p.task_3_training_csv_path, "Task 3 Training Set")
    # training_set_summary(p.task_3_training_path, p.task_3_validation_csv_path, "Task 3 Validation Set")

    # testing_set_summary(p.task_1_testing_path, p.task_1_testing_csv_path, "Task 1 Testing Set")
    # testing_set_summary(p.task_2_testing_path, p.task_2_testing_csv_path, "Task 2 Testing Set")
    # testing_set_summary(p.task_3_testing_path, p.task_3_testing_csv_path, "Task 3 Testing Set")

    # training_set_summary(p.task_1_training_preprocessed_path, p.task_1_training_csv_path, "Task 1 Preprocessed Training Set")
    # training_set_summary(p.task_1_training_preprocessed_path, p.task_1_validation_csv_path, "Task 1 Preprocessed Validation Set")
    # training_set_summary(p.task_3_training_preprocessed_path, p.task_3_training_csv_path, "Task 3 Preprocessed Training Set")
    # training_set_summary(p.task_3_training_preprocessed_path, p.task_3_validation_csv_path, "Task 3 Preprocessed Validation Set")

    # testing_set_summary(p.task_1_testing_preprocessed_path, p.task_1_testing_csv_path, "Task 1 Preprocessed Testing Set")
    # testing_set_summary(p.task_2_testing_preprocessed_path, p.task_2_testing_csv_path, "Task 2 Preprocessed Testing Set")
    # testing_set_summary(p.task_3_testing_preprocessed_path, p.task_3_testing_csv_path, "Task 3 Preprocessed Testing Set")
    pass

if __name__ == "__main__":
    run()