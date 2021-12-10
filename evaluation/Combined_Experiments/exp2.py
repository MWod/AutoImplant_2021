"""
Experiment description:

Evaluation of Task 3 - Exp 1.
"""

import os
import sys
import getopt
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import utils as u
import paths as p
import datasets as ds
import evaluation as ev
import torch.utils.tensorboard as tb
from networks import unet


def evaluation():
    experiment_name = "Combined_Ev2"
    experiments_family = "Combined_Evaluation"
    ### Experiment Parameters Defined Below ###
    device = "cuda:0"
    cost_function = u.dice_loss
    dataset_mode = "defect_implant"
    data_path = p.combined_training_path
    input_csv = p.combined_validation_csv_path
    initial_weights_path = str(p.combined_exp2_save_path / str("model_cp3"))
    dataset_name = "C2_Val"

    log_dir = p.logs_path / experiments_family / experiment_name
    comment = "Combined - Exp2"
    logger = tb.SummaryWriter(log_dir=log_dir, comment=comment)
    ###########################################

    evaluation_params = dict()
    evaluation_params['device'] = device
    evaluation_params['cost_function'] = cost_function
    evaluation_params['dataset_mode'] = dataset_mode
    evaluation_params['dataset_name'] = dataset_name
    evaluation_params['data_path'] = data_path
    evaluation_params['model_file'] = unet
    evaluation_params['initial_weights_path'] = initial_weights_path
    evaluation_params['input_csv'] = input_csv
    evaluation_params['logger'] = logger

    ############################
    ev.evaluation(evaluation_params)
    ############################

if __name__ == "__main__":
    evaluation()