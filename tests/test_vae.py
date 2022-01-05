import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


import numpy as np
import matplotlib.pyplot as plt
import torch as tc

import datasets as ds
import augmentation as aug
import utils as u
import paths as p

from networks import vae
import training_variational_augmentation as va


def test_vae_1():
    device = "cpu"
    data_folder = p.combined_training_path
    csv_file = p.combined_validation_csv_path
    mode = "defect_implant"

    batch_size = 1
    dataloader = ds.create_dataloader(data_folder, csv_file, mode, batch_size, iteration_size=100,
     transforms=aug.generate_transforms_flip_affine(scales=(0.9, 1.1), degrees=(-5, 5)))
    print("Dataloader size: ", len(dataloader.dataset))

    model_path = p.vae_exp5_save_path / "model_cp5"
    model = vae.load_network().to(device)
    state_dict = u.parse_state_dict(str(model_path), keyword="vae.")
    model.load_state_dict(state_dict)
    model.eval()


    for defects, implants, spacings in dataloader:
        print("Defects size: ", defects.size())
        print("Implants size: ", implants.size())
        print("Spacings size: ", spacings.size())

        defect, implant, spacing = defects[0, :, :, :], implants[0, :, :, :], spacings[0]

        print("Defect size: ", defect.size())
        print("Implant size: ", implant.size())
        print("Spacing: ", spacing)

        input, ground_truth = defects, implants
        input = input.view(input.size(0), 1, *input.size()[1:])
        ground_truth = ground_truth.view(ground_truth.size(0), 1, *ground_truth.size()[1:])
        vae_input = tc.cat((input, ground_truth), dim=1)
        vae_output, z, mean, std = model(vae_input)

        vae_defect = vae_output[0, 0, :, :, :].detach().cpu().numpy()
        vae_implant = vae_output[0, 1, :, :, :].detach().cpu().numpy()

        mse = lambda v1, v2: tc.mean((v1-v2)**2)
        reconstruction_loss = mse(vae_input, vae_output)
        distribution_loss = u.kld(z, mean, std)
        print(f"Reconsturction loss: {reconstruction_loss}")
        print(f"Distribution loss:  {distribution_loss}")

        u.show_training_case(None, defect.detach().cpu().numpy(), implant.detach().cpu().numpy(), spacing.numpy(), show=False)
        u.show_training_case(None, vae_defect, vae_implant, spacing.numpy(), show=False)
        plt.show()

def test_vae_2():
    device = "cpu"

    model_path = p.vae_exp5_save_path / "model_cp5"
    model = vae.load_network().to(device)
    state_dict = u.parse_state_dict(str(model_path), keyword="vae.")
    model.load_state_dict(state_dict)
    model.eval()

    no_samples = 20
    for i in range(no_samples):
        vae_output = model.generate()

        vae_defect = vae_output[0, 0, :, :, :].detach().cpu().numpy()
        vae_implant = vae_output[0, 1, :, :, :].detach().cpu().numpy()

        print("Mean: ", tc.mean(vae_output))

        mse = lambda v1, v2: tc.mean((v1-v2)**2)
        # reconstruction_loss = mse(vae_input, vae_output)
        # distribution_loss = va.kld(z, mean, std)
        # print(f"Reconsturction loss: {reconstruction_loss}")
        # print(f"Distribution loss:  {distribution_loss}")

        # u.show_training_case(None, defect.detach().cpu().numpy(), implant.detach().cpu().numpy(), spacing.numpy(), show=False)
        u.show_training_case(np.logical_or(vae_defect, vae_implant), vae_defect, vae_implant, (1.0, 1.0, 1.0), show=False)
        plt.show()

def run():
    # test_vae_1()
    test_vae_2()
    pass


if __name__ == "__main__":
    run()