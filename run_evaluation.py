import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import torch as tc
import matplotlib.pyplot as plt

import utils as u
import paths as p
import evaluation_metrics as metrics

from networks import unet

import pipeline as pp

def run_evaluation_on_validation_set(output_data_path, output_csv_path, reconstruction_params, echo=False):
    """
    Performs the validation on Task 1/3 validation sets.
    """
    dataframe = pd.read_csv(p.task_1_validation_csv_path)
    print("Task 1 Dataset size: ", len(dataframe))

    dcs = []
    bdcs = []
    hds95 = []
    results = []
    for current_id, case in dataframe.iterrows():
        print("Current ID: ", current_id)
        defective_skull_path = p.task_1_training_path / case['Defective Skull Path']
        implant_path = p.task_1_training_path / case['Implant Path']
        to_save_path = output_data_path / "Validation" / "Task1" / case['Defective Skull Path']

        defective_skull, _, _ = u.load_volume(defective_skull_path)
        implant, spacing, _ = u.load_volume(implant_path)

        reconstructed_implant = pp.defect_reconstruction(defective_skull_path, to_save_path, echo=echo, **reconstruction_params)

        dc = metrics.dc(reconstructed_implant, implant)
        bdc = metrics.bdc(reconstructed_implant, implant, defective_skull, spacing)
        hd95 = metrics.hd95(reconstructed_implant, implant, spacing)
        dcs.append(dc)
        bdcs.append(bdc)
        hds95.append(hd95)
        results.append(["Task 1 " + case['Defective Skull Path'], dc, bdc, hd95])
    results.append(["Mean", np.mean(dcs), np.mean(bdcs), np.mean(hds95)])

    dataframe = pd.read_csv(p.task_3_validation_csv_path)
    print("Task 3 Dataset size: ", len(dataframe))

    dcs = []
    bdcs = []
    hds95 = []
    for current_id, case in dataframe.iterrows():
        print("Current ID: ", current_id)
        defective_skull_path = p.task_3_training_path / case['Defective Skull Path']
        implant_path = p.task_3_training_path / case['Implant Path']
        to_save_path = output_data_path / "Validation" / "Task3" / case['Defective Skull Path']

        defective_skull, _, _ = u.load_volume(defective_skull_path)
        implant, spacing, _ = u.load_volume(implant_path)

        reconstructed_implant = pp.defect_reconstruction(defective_skull_path, to_save_path, echo=echo, **reconstruction_params)

        dc = metrics.dc(reconstructed_implant, implant)
        bdc = metrics.bdc(reconstructed_implant, implant, defective_skull, spacing)
        hd95 = metrics.hd95(reconstructed_implant, implant, spacing)
        dcs.append(dc)
        bdcs.append(bdc)
        hds95.append(hd95)
        results.append(["Task 3 " + case['Defective Skull Path'], dc, bdc, hd95])
    results.append(["Mean", np.mean(dcs), np.mean(bdcs), np.mean(hds95)])

    output_results = pd.DataFrame(results, columns=['Case', "Dice", "Boundary Dice", "HD95"])
    output_results.to_csv(output_csv_path, index=False)

def run_evaluation_on_testing_set(output_data_path, reconstruction_params, echo=False, task_1=True, task_2=True, task_3=True):
    """
    Performs the evaluation on Task 1/2/3 test sets.
    """
    if task_1:
        dataframe = pd.read_csv(p.task_1_testing_csv_path)
        print("Task 1 Dataset size: ", len(dataframe))
        for current_id, case in dataframe.iterrows():
            print("Current ID: ", current_id)
            defective_skull_path = p.task_1_testing_path / case['Defective Skull Path']
            first_split = os.path.split(case['Defective Skull Path'])
            second_split = os.path.split(first_split[0])
            to_save = os.path.join(second_split[1], first_split[1])
            to_save_path = output_data_path / "Testing" / "Task1" / to_save
            _ = pp.defect_reconstruction(defective_skull_path, to_save_path, echo=echo, **reconstruction_params)

    if task_2:
        dataframe = pd.read_csv(p.task_2_testing_csv_path)
        print("Task 2 Dataset size: ", len(dataframe))
        for current_id, case in dataframe.iterrows():
            print("Current ID: ", current_id)
            defective_skull_path = p.task_2_testing_path / case['Defective Skull Path']
            to_save_path = output_data_path / "Testing" / "Task2" / case['Defective Skull Path']
            _ = pp.defect_reconstruction(defective_skull_path, to_save_path, echo=echo, **reconstruction_params)

    if task_3:
        dataframe = pd.read_csv(p.task_3_testing_csv_path)
        print("Task 3 Dataset size: ", len(dataframe))
        for current_id, case in dataframe.iterrows():
            print("Current ID: ", current_id)
            defective_skull_path = p.task_3_testing_path / case['Defective Skull Path']
            to_save_path = output_data_path / "Testing" / "Task3" / case['Defective Skull Path']
            _ = pp.defect_reconstruction(defective_skull_path, to_save_path, echo=echo, **reconstruction_params)

def produce_csv_with_results(task_1_results_path, task_3_results_path, task_1_gt_path, task_3_gt_path, output_task_1_path, output_task_3_path, task_1=True, task_3=True):
    if task_1:
        results = []
        dcs = []
        bdcs = []
        hds95 = []
        dataframe = pd.read_csv(p.task_1_testing_csv_path)
        for current_id, case in dataframe.iterrows():
            print("Current ID: ", current_id)
            defective_skull_path = p.task_1_testing_path / case['Defective Skull Path']
            reconstruction_path = task_1_results_path / str(case['Defective Skull Path'].split('defective_skull\\')[1])
            implant_path = task_1_gt_path / str(case['Defective Skull Path'].split('defective_skull\\')[1])

            defective_skull, _, _ = u.load_volume(defective_skull_path)
            reconstructed_implant, _, _ = u.load_volume(reconstruction_path)
            implant, spacing, _ = u.load_volume(implant_path)

            dc = metrics.dc(reconstructed_implant, implant)
            bdc = metrics.bdc(reconstructed_implant, implant, defective_skull, spacing)
            hd95 = metrics.hd95(reconstructed_implant, implant, spacing)
            print(f"DC: {dc}, BDC: {bdc}, HD95: {hd95}")
            dcs.append(dc)
            bdcs.append(bdc)
            hds95.append(hd95)
            results.append([case['Defective Skull Path'], dc, bdc, hd95])

        results.append(["Mean", np.mean(dcs), np.mean(bdcs), np.mean(hds95)])
        results = pd.DataFrame(results, columns=['Case', "Dice", "Boundary Dice", "HD95"])
        results.to_csv(output_task_1_path, index=False)

    if task_3:
        results = []
        dcs = []
        bdcs = []
        hds95 = []
        dataframe = pd.read_csv(p.task_3_testing_csv_path)
        for current_id, case in dataframe.iterrows():
            print("Current ID: ", current_id)
            defective_skull_path = p.task_3_testing_path / case['Defective Skull Path']
            reconstruction_path = task_3_results_path / str(case['Defective Skull Path'])
            implant_path = task_3_gt_path / str(case['Defective Skull Path'])

            defective_skull, _, _ = u.load_volume(defective_skull_path)
            reconstructed_implant, _, _ = u.load_volume(reconstruction_path)
            implant, spacing, _ = u.load_volume(implant_path)

            dc = metrics.dc(reconstructed_implant, implant)
            bdc = metrics.bdc(reconstructed_implant, implant, defective_skull, spacing)
            hd95 = metrics.hd95(reconstructed_implant, implant, spacing)
            print(f"DC: {dc}, BDC: {bdc}, HD95: {hd95}")
            dcs.append(dc)
            bdcs.append(bdc)
            hds95.append(hd95)
            results.append([case['Defective Skull Path'], dc, bdc, hd95])

        results.append(["Mean", np.mean(dcs), np.mean(bdcs), np.mean(hds95)])
        results = pd.DataFrame(results, columns=['Case', "Dice", "Boundary Dice", "HD95"])
        results.to_csv(output_task_3_path, index=False)

