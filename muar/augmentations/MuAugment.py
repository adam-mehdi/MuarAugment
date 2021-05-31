from typing import Union

import torch
from torch import nn
import pytorch_lightning as pl

from muar.transforms import BatchRandAugment
from muar.loss import MixUpCrossEntropy


class MuAugment():
    """
        For each image in the batch, applies `N_COMPS` compositions with `N_TFMS` 
        tranformations, each of magnitude `MAGN`, and for each group of `N_COMPS` 
        augmented images, returns the `N_SELECTED` images with the greatest loss. 
    """
    def __init__(self,
                 rand_augment: BatchRandAugment,
                 N_COMPS: int = 4, 
                 N_SELECTED: int = 2, 
                 device: Union[torch.device, str] = None):
        """
        Args:
            rand_augment (muar.BatchRandAugment): RandAugment pipeline to apply.
            N_COMPS (int): Number of compositions placed on each image. [C in paper]
            N_SELECTED (int): Number of selected compositions for each image. [S in paper]
            device (torch.device, optional): Device to use: cuda or cpu. Inferred if `None`.
        
        Returns:
            S_images (torch.Tensor): (batch_size*S, n_channels, height, width)
            S_targets (torch.LongTensor): (batch_size*N_SELECTED) or 
                                          (batch_size*N_SELECTED, 3) if applied MixUp or CutMix
            
        NOTE: Effective batch size is batch_size*S. Increasing C yields better accuracy.
        """
        super().__init__()

        self.rand_augment = rand_augment
        self.C, self.S = N_COMPS, N_SELECTED

        self.device = device
        if device == None: self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        
    def setup(self,  # ALLOW SCHEDULING OF HYPERPARAMETERS (EDIT THEM VIA KWARGS)
              model: Union[nn.Module, pl.LightningModule]):
        """
        Args:
            model (nn.Module, pl.LightningModule): Model used to calculate loss.
        """
        self.model = model
        self.rand_augment.setup()
        
        self.transform = rand_augment
        self.N, self.M = rand_augment.N_TFMS, rand_augment.MAGN
        self.image_size = rand_augment.image_size
        
        if rand_augment.use_mix != 0: self.loss = MixUpCrossEntropy(reduction=False)
  #      else: self.loss =  cutout_cross_entropy
        
        
    @torch.no_grad()
    def __call__(self, batch):
        """
        Args:
            batch (tuple): tuple of (image batch, target batch).
                image batch: (batch_size, n_channels, height, width)
                target batch: (batch_size, 1) or (batch_size)
        """

        xb,yb = batch
        if xb.device != self.device: xb = xb.to(self.device)
        if yb.device != self.device: yb = yb.to(self.device)

        if len(yb.shape) == 2: yb = yb.squeeze(1)
        BS,N_CHANNELS = xb.shape[:2]
        HEIGHT,WIDTH = self.image_size
        C_images = torch.zeros(self.C, BS, N_CHANNELS, HEIGHT, WIDTH, device=self.device)

        if self.transform.mix is not None:
            C_targets = torch.zeros(self.C, BS, 3, device=self.device, dtype=torch.long)
        else:
            C_targets = torch.zeros(self.C, BS, device=self.device, dtype=torch.long)
        
        for c in range(self.C):
            xbt,ybt = self.transform(xb,yb)
            C_images[c],C_targets[c] = xbt,ybt
        
        preds = [self.model(images) for images in C_images]
        
        loss_tensor = torch.stack([self.loss(pred, C_targets[i]) for i,pred in enumerate(preds)])
        
        S_idxs = loss_tensor.topk(self.S, dim=0).indices
                                   
        S_images = C_images[S_idxs, range(BS)]
        S_images = S_images.view(-1, N_CHANNELS, HEIGHT, WIDTH)

        S_targets = C_targets[S_idxs, range(BS)]
        S_targets = S_targets.view(-1, 3) if self.transform.mix else S_targets.view(-1)
            
        return S_images, S_targets