import os
import numpy as np
import matplotlib.pyplot as plt
import torch as tc
import torchvision.transforms as transforms

import utils as u
import datasets as ds
import evaluation_metrics as metrics
import paths as p
import PIL.Image


def evaluation(evaluation_params):
    """
    Function to perform the evaluation on a given training/validation set (for the DL-based segmentation only)
    """
    device  = evaluation_params['device']
    data_path = evaluation_params['data_path']
    input_csv = evaluation_params['input_csv']
    dataset_mode = evaluation_params['dataset_mode']
    dataset_name = evaluation_params['dataset_name']
    model_file = evaluation_params['model_file']
    initial_weights_path = evaluation_params['initial_weights_path']
    cost_function = evaluation_params['cost_function']
    logger = evaluation_params['logger']

    model = model_file.load_network().to(device)
    state_dict = u.parse_state_dict(initial_weights_path)
    model.load_state_dict(state_dict)
    model.eval()
    dataloader = ds.create_dataloader(data_path, input_csv, dataset_mode, 1, shuffle=False, num_workers=4)
    evaluate_dataset(dataloader, model, cost_function, logger, dataset_name, device=device)

def evaluate_dataset(dataloader, model, cost_function, logger, dataset_name, device="cpu"):
    losses = []
    dcs = []
    bdcs = []
    hds = []
    hds95 = []

    print("Dataset size: ", len(dataloader.dataset))
    current_case = 1

    for batch in dataloader:
        with tc.set_grad_enabled(False):
            input, ground_truth, spacing = batch
            spacing = spacing[0].numpy()
            input = input.view(input.size(0), 1, *input.size()[1:]).to(device)
            ground_truth = ground_truth.view(ground_truth.size(0), 1, *ground_truth.size()[1:]).to(device)
            result = model(input)
            loss = cost_function(result, ground_truth)

            case_np = input.detach().cpu().numpy()[0, 0, :, :, :] > 0.5
            result_np = result.detach().cpu().numpy()[0, 0, :, :, :] > 0.5
            ground_truth_np = ground_truth.detach().cpu().numpy()[0, 0, :, :, :] > 0.5

            dc = metrics.dc(result_np, ground_truth_np)
            bdc = metrics.bdc(result_np, ground_truth_np, case_np, spacing)
            hd = metrics.hd(result_np, ground_truth_np, spacing)
            hd95 = metrics.hd95(result_np, ground_truth_np, spacing)

            metric_text = "DC: " + str(round(dc, 2)) + " BDC: " + str(round(bdc, 2)) + " HD: " + str(round(hd, 2)) + " HD95: " + str(round(hd95, 2))

            buf = u.show_training_case(case_np, result_np, ground_truth_np, spacing, names=["Reconstructed", "Input", "Ground-truth"], show=False, return_buffer=True, suptitle=metric_text)
            image = PIL.Image.open(buf)
            image = transforms.ToTensor()(image).unsqueeze(0)[0]
            logger.add_image(dataset_name.capitalize() + " Current case: " + str(current_case), image, 0)

            print("Loss: ", loss.item())
            print("Dice: ", dc)
            print("Boundary Dice, ", bdc)
            print("HD", hd)
            print("HD 95: ", hd95)

            losses.append(loss.item())
            dcs.append(dc)
            bdcs.append(bdc)
            hds.append(hd)
            hds95.append(hd95)

            current_case += 1

            plt.close()
            plt.clf()
    
    losses = np.array(losses)
    dcs = np.array(dcs)
    bdcs = np.array(bdcs)
    hds = np.array(hds)
    hds95 = np.array(hds95)

    logger.add_text(dataset_name.capitalize() + " Loss mean", str(np.mean(losses)), 0)
    logger.add_text(dataset_name.capitalize() + " Loss std", str(np.std(losses)), 0)
    logger.add_text(dataset_name.capitalize() + " Loss min", str(np.min(losses)), 0)
    logger.add_text(dataset_name.capitalize() + " Loss max", str(np.max(losses)), 0)

    logger.add_text(dataset_name.capitalize() + " Dice mean", str(np.mean(dcs)), 0)
    logger.add_text(dataset_name.capitalize() + " Dice std", str(np.std(dcs)), 0)
    logger.add_text(dataset_name.capitalize() + " Dice min", str(np.min(dcs)), 0)
    logger.add_text(dataset_name.capitalize() + " Dice max", str(np.max(dcs)), 0)

    logger.add_text(dataset_name.capitalize() + " Boundary Dice mean", str(np.mean(bdcs)), 0)
    logger.add_text(dataset_name.capitalize() + " Boundary Dice std", str(np.std(bdcs)), 0)
    logger.add_text(dataset_name.capitalize() + " Boundary Dice min", str(np.min(bdcs)), 0)
    logger.add_text(dataset_name.capitalize() + " Boundary Dice max", str(np.max(bdcs)), 0)

    logger.add_text(dataset_name.capitalize() + " HD mean", str(np.mean(hds)), 0)
    logger.add_text(dataset_name.capitalize() + " HD std", str(np.std(hds)), 0)
    logger.add_text(dataset_name.capitalize() + " HD min", str(np.min(hds)), 0)
    logger.add_text(dataset_name.capitalize() + " HD max", str(np.max(hds)), 0)

    logger.add_text(dataset_name.capitalize() + " HD 95 mean", str(np.mean(hds95)), 0)
    logger.add_text(dataset_name.capitalize() + " HD 95 std", str(np.std(hds95)), 0)
    logger.add_text(dataset_name.capitalize() + " HD 95 min", str(np.min(hds95)), 0)
    logger.add_text(dataset_name.capitalize() + " HD 95 max", str(np.max(hds95)), 0)

    logger.flush()

    print("Mode: ", dataset_name.capitalize())

    print("Loss mean: ", np.mean(losses))
    print("Loss std: ", np.std(losses))
    print("Loss min: ", np.min(losses))
    print("Loss max: ", np.max(losses))
    
    print("Dice mean: ", np.mean(dcs))
    print("Dice std: ", np.std(dcs))
    print("Dice min: ", np.min(dcs))
    print("Dice max: ", np.max(dcs))

    print("Boundary Dice mean: ", np.mean(bdcs))
    print("Boundary Dice std: ", np.std(bdcs))
    print("Boundary Dice min: ", np.min(bdcs))
    print("Boundary Dice max: ", np.max(bdcs))

    print("HD mean: ", np.mean(hds))
    print("HD std: ", np.std(hds))
    print("HD min: ", np.min(hds))
    print("HD max: ", np.max(hds))

    print("HD95 mean: ", np.mean(hds95))
    print("HD95 std: ", np.std(hds95))
    print("HD95 min: ", np.min(hds95))
    print("HD95 max: ", np.max(hds95))

    plt.close()
    plt.clf()


