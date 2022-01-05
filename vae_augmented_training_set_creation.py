import os
import pathlib
import shutil

import numpy as np
import matplotlib.pyplot as plt
import torch as tc
import pandas as pd

import paths as p
import utils as u

from networks import vae


def generate_and_combine(input_data_path, input_training_csv, input_validation_csv, output_data_path, output_training_csv, output_validation_csv, images_to_generate, model, spacing=(1.0, 1.0, 1.0), device="cpu"):
    training_data = []
    validation_data = []

    ### Generate
    for i in range(images_to_generate):
        print(f"Current image: {i + 1} / {images_to_generate}")
        generated_image = model.generate(device=device)

        defective_skull = generated_image[0, 0, :, :, :].detach().cpu().numpy() > 0.5
        implant = generated_image[0, 1, :, :, :].detach().cpu().numpy() > 0.5
        complete_skull = np.logical_or(defective_skull, implant)

        defective_skull_path = output_data_path / str(i) / "defective_skull.nrrd"
        implant_path = output_data_path / str(i) / "implant.nrrd"
        complete_skull_path = output_data_path / str(i) / "complete_skull.nrrd"
        
        pathlib.Path(defective_skull_path).parents[0].mkdir(parents=True, exist_ok=True)
        pathlib.Path(implant_path).parents[0].mkdir(parents=True, exist_ok=True)
        pathlib.Path(complete_skull_path).parents[0].mkdir(parents=True, exist_ok=True)

        u.save_volume(defective_skull, spacing, defective_skull_path)
        u.save_volume(implant , spacing, implant_path)
        u.save_volume(complete_skull, spacing, complete_skull_path)
        training_data.append([complete_skull_path, defective_skull_path, implant_path])

    ### Combine Training
    training_dataframe = pd.read_csv(input_training_csv)
    for _, case in training_dataframe.iterrows():
        input_complete_skull_path = input_data_path / case['Complete Skull Path']
        input_defective_skull_path = input_data_path / case['Defective Skull Path']
        input_implant_path = input_data_path / case['Implant Path']

        output_complete_skull_path = output_data_path / case['Complete Skull Path']
        output_defective_skull_path = output_data_path / case['Defective Skull Path']
        output_implant_path = output_data_path / case['Implant Path']

        pathlib.Path(output_complete_skull_path).parents[0].mkdir(parents=True, exist_ok=True)
        pathlib.Path(output_defective_skull_path).parents[0].mkdir(parents=True, exist_ok=True)
        pathlib.Path(output_implant_path).parents[0].mkdir(parents=True, exist_ok=True)

        shutil.copy(input_complete_skull_path, output_complete_skull_path)
        shutil.copy(input_defective_skull_path, output_defective_skull_path)
        shutil.copy(input_implant_path, output_implant_path)
        training_data.append([case['Complete Skull Path'], case['Defective Skull Path'], case['Implant Path']])


    ### Combine Validation
    validation_dataframe = pd.read_csv(input_validation_csv)
    for _, case in validation_dataframe.iterrows():
        input_complete_skull_path = input_data_path / case['Complete Skull Path']
        input_defective_skull_path = input_data_path / case['Defective Skull Path']
        input_implant_path = input_data_path / case['Implant Path']

        output_complete_skull_path = output_data_path / case['Complete Skull Path']
        output_defective_skull_path = output_data_path / case['Defective Skull Path']
        output_implant_path = output_data_path / case['Implant Path']

        pathlib.Path(output_complete_skull_path).parents[0].mkdir(parents=True, exist_ok=True)
        pathlib.Path(output_defective_skull_path).parents[0].mkdir(parents=True, exist_ok=True)
        pathlib.Path(output_implant_path).parents[0].mkdir(parents=True, exist_ok=True)

        shutil.copy(input_complete_skull_path, output_complete_skull_path)
        shutil.copy(input_defective_skull_path, output_defective_skull_path)
        shutil.copy(input_implant_path, output_implant_path)
        validation_data.append([case['Complete Skull Path'], case['Defective Skull Path'], case['Implant Path']])

    training_dataframe = pd.DataFrame(training_data, columns=['Complete Skull Path', "Defective Skull Path", "Implant Path"])
    training_dataframe.to_csv(output_training_csv, index=False)

    validation_dataframe = pd.DataFrame(validation_data, columns=['Complete Skull Path', "Defective Skull Path", "Implant Path"])
    validation_dataframe.to_csv(output_validation_csv, index=False)



def run():
    input_data_path = p.combined_training_path
    output_data_path = p.combined_vae_training_path

    input_training_csv = p.combined_training_csv_path
    output_training_csv = p.combined_vae_training_csv_path
    
    input_validation_csv = p.combined_validation_csv_path
    output_validation_csv = p.combined_vae_validation_csv_path
    
    images_to_generate = 100000

    device = tc.device("cuda:0")
    model_path = p.vae_exp5_save_path / "model_cp5"
    model = vae.load_network().to(device)
    state_dict = u.parse_state_dict(str(model_path), keyword="vae.")
    model.load_state_dict(state_dict)
    model.eval()

    generate_and_combine(input_data_path, input_training_csv, input_validation_csv, output_data_path, output_training_csv, output_validation_csv, images_to_generate, model, device=device)

if __name__ == "__main__":
    run()