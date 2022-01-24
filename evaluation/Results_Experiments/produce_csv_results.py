"""
Experiment description:

Final evaluation of Second Step - Exp 2
"""

import os
import sys
import getopt
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import utils as u
import paths as p
from networks import unet

import run_evaluation as re

def evaluation_1():
    task_1_results_path = p.submissions_path / "lWM_task1_creg"
    task_3_results_path = p.submissions_path / "lWM_task3_creg"
    task_1_gt_path = p.task_1_gt_path
    task_3_gt_path = p.task_3_gt_path
    output_task_1_path = p.results_csv_path / "task1_creg.csv"
    output_task_3_path = p.results_csv_path / "task3_creg.csv"
    re.produce_csv_with_results(task_1_results_path, task_3_results_path, task_1_gt_path, task_3_gt_path, output_task_1_path, output_task_3_path)

def evaluation_2():
    task_1_results_path = p.submissions_path / "lWM_task1_cregref"
    task_3_results_path = p.submissions_path / "lWM_task3_cregref"
    task_1_gt_path = p.task_1_gt_path
    task_3_gt_path = p.task_3_gt_path
    output_task_1_path = p.results_csv_path / "task1_cregref.csv"
    output_task_3_path = p.results_csv_path / "task3_cregref.csv"
    re.produce_csv_with_results(task_1_results_path, task_3_results_path, task_1_gt_path, task_3_gt_path, output_task_1_path, output_task_3_path)

def evaluation_3():
    task_1_results_path = p.submissions_path / "lWM_task1_cvae"
    task_3_results_path = p.submissions_path / "lWM_task3_cvae"
    task_1_gt_path = p.task_1_gt_path
    task_3_gt_path = p.task_3_gt_path
    output_task_1_path = p.results_csv_path / "task1_cvae.csv"
    output_task_3_path = p.results_csv_path / "task3_cvae.csv"
    re.produce_csv_with_results(task_1_results_path, task_3_results_path, task_1_gt_path, task_3_gt_path, output_task_1_path, output_task_3_path)

def evaluation_4():
    task_1_results_path = p.submissions_path / "lWM_task1_cvaeref"
    task_3_results_path = p.submissions_path / "lWM_task3_cvaeref"
    task_1_gt_path = p.task_1_gt_path
    task_3_gt_path = p.task_3_gt_path
    output_task_1_path = p.results_csv_path / "task1_cvaeref.csv"
    output_task_3_path = p.results_csv_path / "task3_cvaeref.csv"
    re.produce_csv_with_results(task_1_results_path, task_3_results_path, task_1_gt_path, task_3_gt_path, output_task_1_path, output_task_3_path)

def evaluation_5():
    task_1_results_path = p.submissions_path / "lWM_task1_cwreg"
    task_3_results_path = p.submissions_path / "lWM_task3_cwreg"
    task_1_gt_path = p.task_1_gt_path
    task_3_gt_path = p.task_3_gt_path
    output_task_1_path = p.results_csv_path / "task1_cwreg.csv"
    output_task_3_path = p.results_csv_path / "task3_cwreg.csv"
    re.produce_csv_with_results(task_1_results_path, task_3_results_path, task_1_gt_path, task_3_gt_path, output_task_1_path, output_task_3_path)

def evaluation_6():
    task_1_results_path = p.submissions_path / "lWM_task1_t1"
    task_3_results_path = p.submissions_path / "lWM_task3_t1"
    task_1_gt_path = p.task_1_gt_path
    task_3_gt_path = p.task_3_gt_path
    output_task_1_path = p.results_csv_path / "task1_t1.csv"
    output_task_3_path = p.results_csv_path / "task3_t1.csv"
    re.produce_csv_with_results(task_1_results_path, task_3_results_path, task_1_gt_path, task_3_gt_path, output_task_1_path, output_task_3_path)

def evaluation_7():
    task_1_results_path = p.submissions_path / "lWM_task1_t3"
    task_3_results_path = p.submissions_path / "lWM_task3_t3"
    task_1_gt_path = p.task_1_gt_path
    task_3_gt_path = p.task_3_gt_path
    output_task_1_path = p.results_csv_path / "task1_t3.csv"
    output_task_3_path = p.results_csv_path / "task3_t3.csv"
    re.produce_csv_with_results(task_1_results_path, task_3_results_path, task_1_gt_path, task_3_gt_path, output_task_1_path, output_task_3_path)


if __name__ == "__main__":
    # evaluation_1()
    # evaluation_2()
    # evaluation_3()
    evaluation_4()
    evaluation_5()
    evaluation_6()
    evaluation_7()
