"""
Experiment description:

Training for Task 3 only (with only geometric augmentation).
"""

import os
import sys
import getopt
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import utils as u
import paths as p
import datasets as ds
import training_implant_reconstruction as tr
import augmentation as aug
import pytorch_lightning as pl


def experiment(current_run):
    current_run = str(current_run)

    experiment_name = "Task3_Exp1"
    experiments_family = "Task3_Experiments"
    ### Experiment Parameters Defined Below ###
    gpus = [0]
    num_workers = 4
    num_iters = 100
    cases_per_iter = 500
    learning_rate = 0.003
    decay_rate = 0.99
    batch_size = 2
    cost_function = u.dice_loss
    transforms = aug.generate_transforms_flip_affine(scales=(0.97, 1.03), degrees=(-6, 6), translation=(-5, 5))
    training_mode = "defect_implant"
    checkpoints_path = p.checkpoints_path / experiments_family / experiment_name
    to_load_checkpoint = None if int(current_run) <= 1 else p.checkpoints_path / experiments_family / experiment_name / str("cp" + str(int(current_run) - 1))
    to_save_checkpoint = p.checkpoints_path / experiments_family / experiment_name / str("cp" + str(current_run))
    data_path = p.task_3_training_preprocessed_path
    training_csv = p.task_3_training_csv_path
    validation_csv = p.task_3_validation_csv_path
    model_save_path = p.task3_exp1_save_path / str("model_cp" + current_run)

    log_dir = p.logs_path / experiments_family / experiment_name / str("cp_" + current_run)
    comment = "Training for Task 1 only (with only geometric augmentation)."
    logger = pl.loggers.TensorBoardLogger(save_dir=log_dir, comment=comment)
    ###########################################

    training_params = dict()
    training_params['gpus'] = gpus
    training_params['num_workers'] = num_workers
    training_params['num_iters'] = num_iters
    training_params['cases_per_iter'] = cases_per_iter
    training_params['learning_rate'] = learning_rate
    training_params['decay_rate'] = decay_rate
    training_params['batch_size'] = batch_size
    training_params['cost_function'] = cost_function
    training_params['transforms'] = transforms
    training_params['training_mode'] = training_mode
    training_params['checkpoints_path'] = checkpoints_path
    training_params['to_load_checkpoint'] = to_load_checkpoint
    training_params['to_save_checkpoint'] = to_save_checkpoint
    training_params['data_path'] = data_path
    training_params['training_csv'] = training_csv
    training_params['validation_csv'] = validation_csv
    training_params['model_save_path'] = model_save_path
    training_params['logger'] = logger

    ############################
    tr.training(training_params)
    ############################

def run(arguments):
    try:
        opts, args = getopt.getopt(arguments ,"c:",)
    except getopt.GetoptError:
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-c":
            current_run = str(arg)
    try:
        if current_run == current_run:
            pass
    except:
        current_run = str(1)
    experiment(current_run)

if __name__ == "__main__":
    experiment(1)
    experiment(2)
    experiment(3)
    pass

