import pathlib
import torch as tc
import pytorch_lightning as pl
import pytorch_lightning.callbacks as cl

from networks import unet
from networks import unet_hpc

import datasets as ds
import utils as u

class ImplantReconstructionDataModule(pl.LightningDataModule):
    def __init__(self, data_path, training_csv, validation_csv, training_mode, batch_size, transforms, num_workers, iteration_size):
        super().__init__()
        self.data_path = data_path
        self.training_csv = training_csv
        self.validation_csv = validation_csv
        self.training_mode = training_mode
        self.batch_size = batch_size
        self.transforms = transforms
        self.num_workers = num_workers
        self.iteration_size = iteration_size

    def train_dataloader(self):
        dataloader = ds.create_dataloader(self.data_path, self.training_csv, self.training_mode, batch_size=self.batch_size, transforms=None,
            num_workers=self.num_workers, iteration_size=self.iteration_size, shuffle=True)
        print()
        print("Training dataloader created, size: ", len(dataloader.dataset))
        print()
        return dataloader

    def val_dataloader(self):
        dataloader = ds.create_dataloader(self.data_path, self.validation_csv, self.training_mode, batch_size=self.batch_size, transforms=None,
            num_workers=self.num_workers, shuffle=False)
        print()
        print("Validation dataloader created, size: ", len(dataloader.dataset))
        print()
        return dataloader

class UNetModule(pl.LightningModule):
    def __init__(self, weights_path, learning_rate, decay_rate, cost_function, use_hpc=False):
        super().__init__()
        if use_hpc:
            self.unet = unet_hpc.load_network(weights_path=weights_path)
        else:
            self.unet = unet.load_network(weights_path=weights_path)
        self.cost_function = cost_function
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate

    def forward(self, x):
        return self.unet(x)

    def configure_optimizers(self):
        optimizer = tc.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = tc.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: self.decay_rate**epoch)
        dict = {'optimizer': optimizer, 'lr_scheduler': scheduler}
        return dict

    def training_step(self, train_batch, batch_idx):
        input, ground_truth, _ = train_batch
        input = input.view(input.size(0), 1, *input.size()[1:])
        ground_truth = ground_truth.view(ground_truth.size(0), 1, *ground_truth.size()[1:])
        result = self.unet(input)
        loss = self.cost_function(result, ground_truth)
        self.log("Training_Loss", loss, prog_bar=True)
        return loss

    def validation_step(self, validation_batch, batch_idx):
        input, ground_truth, _ = validation_batch
        input = input.view(input.size(0), 1, *input.size()[1:])
        ground_truth = ground_truth.view(ground_truth.size(0), 1, *ground_truth.size()[1:])
        result = self.unet(input)
        loss = self.cost_function(result, ground_truth)
        self.log("Validation_Loss", loss, prog_bar=True, sync_dist=True)

def training(training_params):
    gpus = training_params['gpus']
    num_workers = training_params['num_workers']
    num_iters = training_params['num_iters']
    cases_per_iter = training_params['cases_per_iter']
    learning_rate = training_params['learning_rate']
    decay_rate = training_params['decay_rate']
    batch_size = training_params['batch_size']
    cost_function = training_params['cost_function']
    transforms = training_params['transforms']
    training_mode = training_params['training_mode']
    logger = training_params['logger']
    checkpoints_path = training_params['checkpoints_path']
    to_load_checkpoint = training_params['to_load_checkpoint']
    to_save_checkpoint = training_params['to_save_checkpoint']
    try:
        save_best = training_params['save_best']
    except KeyError:
        save_best = False
    try:
        use_hpc = training_params['use_hpc']
    except KeyError:
        use_hpc = False

    data_path = training_params['data_path']
    training_csv = training_params['training_csv']
    validation_csv = training_params['validation_csv']

    model_save_path = training_params['model_save_path']

    data_module = ImplantReconstructionDataModule(data_path, training_csv, validation_csv, training_mode, batch_size, transforms=transforms,
        num_workers=num_workers, iteration_size=cases_per_iter)

    if to_load_checkpoint is None:
        model = UNetModule(None, learning_rate, decay_rate, cost_function, use_hpc=use_hpc)
    else:
        model = UNetModule.load_from_checkpoint(checkpoint_path=str(to_load_checkpoint), weights_path=None, learning_rate=learning_rate, decay_rate=decay_rate, cost_function=cost_function, use_hpc=use_hpc)

    if not save_best:
        trainer = pl.Trainer(gpus=gpus, logger=logger, max_epochs=num_iters, reload_dataloaders_every_n_epochs=1, default_root_dir=checkpoints_path)
        trainer.fit(model, data_module)
        trainer.save_checkpoint(str(to_save_checkpoint))
    else:
        checkpoint_callback = cl.ModelCheckpoint(monitor="Validation_Loss", dirpath=checkpoints_path, filename="Best_Checkpoint",
            save_top_k=3, mode="min")
        trainer = pl.Trainer(gpus=gpus, logger=logger, max_epochs=num_iters, reload_dataloaders_every_n_epochs=1, default_root_dir=checkpoints_path, callbacks=[checkpoint_callback])
        trainer.fit(model, data_module)
        trainer.save_checkpoint(str(to_save_checkpoint))
        model = UNetModule.load_from_checkpoint(checkpoint_path=str(checkpoint_callback.best_model_path), weights_path=None, learning_rate=learning_rate, decay_rate=decay_rate, cost_function=cost_function, use_hpc=use_hpc)

    if model_save_path is not None:
        pathlib.Path(model_save_path).parents[0].mkdir(parents=True, exist_ok=True)
        tc.save(model.state_dict(), str(model_save_path))
