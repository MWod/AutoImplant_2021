import os
import numpy as np
import pandas as pd
import torch as tc
import torch.utils as tcu

import utils as u

import augmentation as aug

def collate_to_list_defect(batch):
    """
    Utility function to create a list of cases for images with different resolutions (only defective cases)
    """
    defect = [item[0].view(item[0].size(0), item[0].size(1), item[0].size(2)) for item in batch]
    spacing = [item[1] for item in batch]
    return defect, spacing

def collate_to_list_both(batch):
    """
    Utility function to create a list of cases for images with different resolutions (defective case + ground-truth)
    """
    defect = [item[0].view(item[0].size(0), item[0].size(1), item[0].size(2)) for item in batch]
    other = [item[1].view(item[1].size(0), item[1].size(1), item[1].size(2)) for item in batch]
    spacing = [item[2] for item in batch]
    return defect, other, spacing

class AutoImplantDataset(tcu.data.Dataset):
    """
    Dataset manager dedicated to the AutoImplant 2021 challenge.
    """
    def __init__(self, data_folder, csv_file, mode, iteration_size=-1, transforms=None, dtype=tc.uint8):
        """
        dataset_folder - path to the folder being the relative paths to paths in csv_file
        csv_file - the dataset file (see dataset_creator.py)
        mode - whether to load only defects, defective skull and implant, or defective and complete skull (defect, defect_complete, defect_implant respectively)
        dtype - desired tensor datatype
        """
        self.data_folder = data_folder
        self.csv_file = csv_file
        self.iteration_size = iteration_size
        self.df = pd.read_csv(self.csv_file)
        if self.iteration_size > 0 and self.iteration_size < len(self.df):
            self.df = self.df.sample(frac=1).reset_index(drop=True)
        self.mode = mode
        self.dtype = dtype
        self.transforms = transforms

    def __len__(self):
        if self.iteration_size > 0 and self.iteration_size < len(self.df):
            return self.iteration_size
        else:
            return len(self.df)

    def __getitem__(self, idx):
        current_case = self.df.loc[idx]
        defect_path = current_case['Defective Skull Path']
        if self.mode == "defect":
            defect, spacing, _ = u.load_volume(self.data_folder / defect_path)
            if self.transforms is not None:
                defect = aug.apply_transform(defect, self.transforms)
            spacing = tc.Tensor(spacing)
            return tc.from_numpy(defect).to(self.dtype), spacing
        if self.mode == "defect_complete":
            complete_path = current_case['Complete Skull Path']
            defect, spacing, _ = u.load_volume(self.data_folder / defect_path)
            complete, _, _ = u.load_volume(self.data_folder / complete_path)
            if self.transforms is not None:
                defect, complete = aug.apply_transform(defect, complete, self.transforms)
            spacing = tc.Tensor(spacing)
            return tc.from_numpy(defect).to(self.dtype), tc.from_numpy(complete).to(self.dtype), spacing
        if self.mode == "defect_implant":
            implant_path = current_case['Implant Path']
            defect, spacing, _ = u.load_volume(self.data_folder / defect_path)
            implant, _, _ = u.load_volume(self.data_folder / implant_path)
            if self.transforms is not None:
                defect, implant = aug.apply_transform(defect, implant, self.transforms)
            spacing = tc.Tensor(spacing)
            return tc.from_numpy(defect).to(self.dtype), tc.from_numpy(implant).to(self.dtype), spacing

def create_dataloader(data_folder, csv_path, mode, batch_size, collate_fn=None, transforms=None, num_workers=8, shuffle=False, iteration_size=-1, dtype=tc.float32):
    """
    Utility function to create dataloader using the AutoImplant dataset.
    """
    dataset = AutoImplantDataset(data_folder, csv_path, mode, iteration_size=iteration_size, transforms=transforms, dtype=dtype)
    dataloader = tcu.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, num_workers=num_workers)
    return dataloader

if __name__ == "__main__":
    pass