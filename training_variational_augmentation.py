import pathlib
import torch as tc
import pytorch_lightning as pl
import pytorch_lightning.callbacks as cl
from torch.distributions import distribution

import matplotlib.pyplot as plt

from networks import vae

import datasets as ds
import utils as u

from training_implant_reconstruction import ImplantReconstructionDataModule

class VAEModule(pl.LightningModule):
    def __init__(self, weights_path, learning_rate, decay_rate, reconstruction_loss, distribution_loss, dist_weight=1.0, use_hpc=False, use_logscale=True):
        super().__init__()
        if use_hpc:
            self.vae = None # TO DO
        else:
            self.vae = vae.load_network(weights_path=weights_path, use_logscale=use_logscale)
        self.reconstruction_loss = reconstruction_loss
        self.distribution_loss = distribution_loss
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.dist_weight = dist_weight
        self.domain_loss = lambda input: -u.dice_loss(input[:, 0, :, :, :], input[:, 1, :, :, :])

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
        vae_input = tc.cat((input, ground_truth), dim=1)
        vae_output, z, mean, std = self.vae(vae_input)

        domain_loss = self.domain_loss(vae_output)
        reconstruction_loss = self.reconstruction_loss(vae_input, vae_output)
        distribution_loss = self.distribution_loss(z, mean, std)
        loss = (reconstruction_loss + self.dist_weight*distribution_loss + domain_loss) / 3.0
        self.log("Training_Reconstruction_Loss", reconstruction_loss, prog_bar=False)
        self.log("Training_Distribution_Loss", distribution_loss, prog_bar=False)
        self.log("Training_Domain_Loss", domain_loss, prog_bar=False)
        self.log("Training_Loss", loss, prog_bar=True)
        return loss

    def validation_step(self, validation_batch, batch_idx):
        input, ground_truth, _ = validation_batch
        input = input.view(input.size(0), 1, *input.size()[1:])
        ground_truth = ground_truth.view(ground_truth.size(0), 1, *ground_truth.size()[1:])
        vae_input = tc.cat((input, ground_truth), dim=1)
        vae_output, z, mean, std = self.vae(vae_input)

        domain_loss = self.domain_loss(vae_output)
        reconstruction_loss = self.reconstruction_loss(vae_input, vae_output)
        distribution_loss = self.distribution_loss(z, mean, std)
        loss = (reconstruction_loss + self.dist_weight*distribution_loss + domain_loss) / 3.0
        self.log("Validation_Reconstruction_Loss", reconstruction_loss, prog_bar=False)
        self.log("Validation_Distribution_Loss", distribution_loss, prog_bar=False)
        self.log("Validation_Domain_Loss", domain_loss, prog_bar=False)
        self.log("Validation_Loss", loss, prog_bar=True)

def training(training_params):
    gpus = training_params['gpus']
    num_workers = training_params['num_workers']
    num_iters = training_params['num_iters']
    cases_per_iter = training_params['cases_per_iter']
    learning_rate = training_params['learning_rate']
    decay_rate = training_params['decay_rate']
    batch_size = training_params['batch_size']
    transforms = training_params['transforms']
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
    try:
        dist_weight = training_params['dist_weight']
    except KeyError:
        dist_weight = 1.0

    data_path = training_params['data_path']
    training_csv = training_params['training_csv']
    validation_csv = training_params['validation_csv']

    model_save_path = training_params['model_save_path']

    data_module = ImplantReconstructionDataModule(data_path, training_csv, validation_csv, "defect_implant", batch_size, transforms=transforms,
        num_workers=num_workers, iteration_size=cases_per_iter)

    reconstruction_loss = u.dice_loss_multichannel
    distribution_loss = u.kld

    if to_load_checkpoint is None:
        model = VAEModule(None, learning_rate, decay_rate, reconstruction_loss, distribution_loss, dist_weight=dist_weight, use_hpc=use_hpc)
    else:
        model = VAEModule.load_from_checkpoint(checkpoint_path=str(to_load_checkpoint), weights_path=None, learning_rate=learning_rate, decay_rate=decay_rate, reconstruction_loss=reconstruction_loss, distribution_loss=distribution_loss, dist_weight=dist_weight, use_hpc=use_hpc)

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
        model = VAEModule.load_from_checkpoint(checkpoint_path=str(checkpoint_callback.best_model_path), weights_path=None, learning_rate=learning_rate, decay_rate=decay_rate, reconstruction_loss=reconstruction_loss, distribution_loss=distribution_loss, dist_weight=dist_weight, use_hpc=use_hpc)

    if model_save_path is not None:
        pathlib.Path(model_save_path).parents[0].mkdir(parents=True, exist_ok=True)
        tc.save(model.state_dict(), str(model_save_path))
