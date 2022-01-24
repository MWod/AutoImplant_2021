import os
import pathlib

### Paths to be changed ###

task_1_path = pathlib.PurePath(r'E:\Data\AutoImplant\Task1')
task_2_path = pathlib.PurePath(r'E:\Data\AutoImplant\Task2')
task_3_path = pathlib.PurePath(r'E:\Data\AutoImplant\Task3')
combined_path = pathlib.PurePath(r'E:\Data\AutoImplant\Combined')
combined_vae_path = pathlib.PurePath(r'E:\Data\AutoImplant\Combined_VAE')
second_step_path = pathlib.PurePath(r'E:\Data\AutoImplant\Second_Step')
results_path = pathlib.PurePath(r'E:\Data\AutoImplant\Results')
stl_path = pathlib.PurePath(r'E:\Data\AutoImplant\STL')

models_path = pathlib.PurePath(r'D:\Research\AutoImplant\Code\models')
logs_path = pathlib.PurePath(r'D:\Research\AutoImplant\Code\logs')
checkpoints_path = pathlib.PurePath(r'D:\Research\AutoImplant\Code\checkpoints')

task_1_gt_path = pathlib.PurePath(r'E:\Data\AutoImplant\Test_Set_Ground_Truth\Task1')
task_3_gt_path = pathlib.PurePath(r'E:\Data\AutoImplant\Test_Set_Ground_Truth\Task3')
submissions_path = pathlib.PurePath(r'E:\Data\AutoImplant\Submissions')

### Paths to not be changed ###

### Dataset paths ###

task_1_training_path = task_1_path / "Training"
task_1_training_preprocessed_path = task_1_path / "Training_Preprocessed"
task_1_testing_path = task_1_path / "Testing"
task_1_testing_preprocessed_path = task_1_path / "Testing_Preprocessed"

task_2_testing_path = task_2_path / "Testing"
task_2_testing_preprocessed_path = task_2_path / "Testing_Preprocessed"

task_3_training_path = task_3_path / "Training"
task_3_training_preprocessed_path = task_3_path / "Training_Preprocessed"
task_3_testing_path = task_3_path / "Testing"
task_3_testing_preprocessed_path = task_3_path / "Testing_Preprocessed"

combined_training_path = combined_path / "Training"

second_step_training_path = second_step_path / "Training"
second_step_implant_training_path = second_step_path / "Training_Implant"

combined_vae_training_path = combined_vae_path / "Training"


### Results paths ###

task1_exp1_results_path = results_path / "Task1_Exp1"
task3_exp1_results_path = results_path / "Task3_Exp1"
combined_exp2_results_path = results_path / "Combined_Exp2"
combined_exp3_results_path = results_path / "Combined_Exp3"
combined_vae_exp1_results_path = results_path / "Combined_VAE_Exp1"
second_step_exp2_results_path = results_path / "Second_Step_Exp2"
second_step_exp2a_results_path = results_path / "Second_Step_Exp2a"
second_step_exp3_results_path = results_path / "Second_Step_Exp3"
second_step_exp3a_results_path = results_path / "Second_Step_Exp3a"
second_step_exp4_results_path = results_path / "Second_Step_Exp4"
implant_modeling_exp1_results_path = results_path / "Implant_Modeling_Exp1"
implant_modeling_exp2_results_path = results_path / "Implant_Modeling_Exp2"
implant_modeling_exp3_results_path = results_path / "Implant_Modeling_Exp3"
implant_modeling_exp4_results_path = results_path / "Implant_Modeling_Exp4"
results_csv_path = results_path / "CSV"

### CSV paths ###

task_1_dataset_csv_path = task_1_path / "task1.csv"
task_1_training_csv_path = task_1_path / "task1_training.csv"
task_1_validation_csv_path = task_1_path / "task1_validation.csv"
task_1_testing_csv_path = task_1_path / "task1_testing.csv"

task_2_dataset_csv_path = task_2_path / "task2.csv"
task_2_testing_csv_path = task_2_path / "task2_testing.csv"

task_3_dataset_csv_path = task_3_path / "task3.csv"
task_3_training_csv_path = task_3_path / "task3_training.csv"
task_3_validation_csv_path = task_3_path / "task3_validation.csv"
task_3_testing_csv_path = task_3_path / "task3_testing.csv"

combined_training_csv_path = combined_path / "combined_training.csv"
combined_validation_csv_path = combined_path / "combined_validation.csv"
combined_training_wr_csv_path = combined_path / "combined_training_wr.csv"

second_step_training_csv_path = second_step_path / "second_step_training.csv"
second_step_validation_csv_path = second_step_path / "second_step_validation.csv"

combined_vae_training_csv_path = combined_vae_path / "combined_vae_training.csv"
combined_vae_validation_csv_path = combined_vae_path / "combined_vae_validation.csv"

### Experiments and models paths ###
simple_exp1_save_path = models_path / "Simple_Experiments"

# Task 1
task1_exp1_save_path = models_path / "Task1_Experiments" / "Task1_Exp1"

# Task 3
task3_exp1_save_path = models_path / "Task3_Experiments" / "Task3_Exp1"

# Combined
combined_exp1_save_path = models_path / "Combined_Experiments" / "Combined_Exp1"
combined_exp2_save_path = models_path / "Combined_Experiments" / "Combined_Exp2"
combined_exp3_save_path = models_path / "Combined_Experiments" / "Combined_Exp3"

# Combined HPC
combined_hpc_exp1_save_path = models_path / "Combined_HPC_Experiments" / "Combined_HPC_Exp1"

# Second Step
second_step_exp1_save_path = models_path / "Second_Step_Experiments" / "Second_Step_Exp1"
second_step_exp2_save_path = models_path / "Second_Step_Experiments" / "Second_Step_Exp2"
second_step_exp3_save_path = models_path / "Second_Step_Experiments" / "Second_Step_Exp3"

# Combined VAE
combined_vae_exp1_save_path = models_path / "Combined_VAE_Experiments" / "Combined_VAE_Exp1"

# VAE
vae_exp1_save_path = models_path / "VAE_Experiments" / "VAE_Exp1"
vae_exp2_save_path = models_path / "VAE_Experiments" / "VAE_Exp2"
vae_exp3_save_path = models_path / "VAE_Experiments" / "VAE_Exp3"
vae_exp4_save_path = models_path / "VAE_Experiments" / "VAE_Exp4"
vae_exp5_save_path = models_path / "VAE_Experiments" / "VAE_Exp5"

