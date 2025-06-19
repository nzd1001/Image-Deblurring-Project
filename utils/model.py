from utils.ssim_loss import SSIMLoss
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl

class DeblurAdvancedUnet(pl.LightningModule):
    def __init__(self, model, lr=1e-4, use_combined_loss=True):
        super().__init__()
        self.model = model
        self.lr = lr
        self.use_combined_loss = use_combined_loss
        self.mse_loss = nn.MSELoss()
        self.ssim_loss=SSIMLoss()
        self.w1 = 0.001  # Trade-off parameter for SSIM

    def compute_loss(self, preds, targets):
        mse = self.mse_loss(preds, targets)
        ssim_loss = self.ssim_loss(preds, targets)
        if self.use_combined_loss:
            return mse + self.w1 * ssim_loss
        return mse
    def forward(self, x):
        return self.model(x)
    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        preds = self.model(inputs)
        preds=torch.clamp(preds,0.0,1.0)
        loss = self.compute_loss(preds, targets)
        self.log("train_loss", loss, prog_bar=True,on_epoch=True,on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        preds = self.model(inputs)
        preds=torch.clamp(preds,0.0,1.0)
        loss = self.compute_loss(preds, targets)
        self.log("val_loss", loss, prog_bar=True,on_epoch=True,on_step=False)
        return loss

    def configure_optimizers(self):
        optimizer=torch.optim.Adam(self.parameters(), lr=self.lr,betas=(0.9, 0.999))
        #scheduler = StepLR(optimizer, step_size=500, gamma=0.5)
        return [optimizer]#,[scheduler]