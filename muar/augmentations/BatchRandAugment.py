from typing import Union
from random import random

import numpy as np
import torch
from torch import nn

import kornia.augmentation as K
from muar.transform_lists import kornia_list



class BatchRandAugment(nn.Module):
    def __init__(self, 
                 N_TFMS: int, 
                 MAGN: int, 
                 mean: Union[tuple, list, torch.tensor],
                 std: Union[tuple, list, torch.tensor],
                 transform_list: list = None, 
                 use_resize: int = None,
                 image_size: tuple = None,
                 use_mix: int = None,
                 mix_p: float = .5):
        """
        Image augmentation pipeline that applies a composition of `N_TFMS` transforms 
        each of magnitude `MAGN` sampled uniformly at random from `transform_list` with 
        optional batch resizing and label mixing transforms.
        
        Args:
            N_TFMS (int): Number of transformations sampled for each composition,
                          excluding resize or label mixing transforms.      <N in paper>
            MAGN (int): Magnitude of augmentation applied. Ranges from [0, 10] with 
                        10 being the max magnitude.                         <M in paper>
            mean (tuple, torch.Tensor): Mean of images after normalized in range [0,1]
            std (tuple, torch.Tensor): Mean of images after normalized in range [0,1]
            transform_list (list): List of transforms to sample from. Default list
                                   provided if not specified.
            use_resize (int): Batch-wise resize transform to apply. Options:
                None: Don't use.
                0: RandomResizedCrop
                1: RandomCrop
                2: CenterCrop
                3: Randomly select a resize transform per batch.
            image_size (tuple): Final size after applying batch-wise resize transforms.
            use_mix (int): Label mixing transform to apply. Options:
                None: Don't use.
                0: CutMix
                1: MixUp
            mix_p (float): probability of applying the mix transform on a batch
                           given `use_mix` is not None.
        """
        super().__init__()
        
        self.N_TFMS, self.MAGN = N_TFMS, MAGN
        self.use_mix, self.mix_p = use_mix, mix_p
        self.image_size = image_size
        
        if not isinstance(mean, torch.Tensor): mean = torch.Tensor(mean)
        if not isinstance(std, torch.Tensor): std = torch.Tensor(std)
            
        if self.use_mix is not None:
            self.mix_list = [K.RandomCutMix(self.image_size[0], self.image_size[1], p=1), 
                             K.RandomMixUp(p=1)]

        self.use_resize = use_resize
        if use_resize is not None:
            assert len(image_size) == 2, 'Invalid `image_size`. Must be a tuple of form (h, w)'
            self.resize_list = [K.RandomResizedCrop(image_size),
                                K.RandomCrop(image_size),
                                K.CenterCrop(image_size)]
            if self.use_resize < 3:
                self.resize = self.resize_list[use_resize]
            
        self.normalize = K.Normalize(mean, std)

        self.transform_list = transform_list    
        if transform_list == None:
            self.transform_list = kornia_list(MAGN)
        
    def setup(self):
        if self.use_resize == 3:
                self.resize = np.random.choice(self.resize_list)
        
        if self.use_mix is not None and random() < self.mix_p:
            self.mix = self.mix_list[self.use_mix]
        else: 
            self.mix = None
            
        sampled_tfms = list(np.random.choice(self.transform_list, self.N_TFMS, replace=False)) + [self.normalize]
        self.transform = nn.Sequential(*sampled_tfms)

        
        
    @torch.no_grad()
    def forward(self, x: torch.Tensor, y: torch.Tensor=None):
        """
        Applies transforms on the batch. 
        
        If `use_mix` is not `None`, `y` is required. Else, it is optional.
        
        If a label-mixing transform is applied on the batch, `y` is returned 
        in shape `(batch_size, 3)`, in which case use the special loss function 
        provided in `muar.utils` if.
        
        Args:
            x (torch.Tensor): Batch of input images.
            y (torch.Tensor): Batch of labels.
        """
        if self.use_resize is not None: 
            x = self.resize(x)

        if self.mix is not None: 
            x,y = self.mix(x, y)
            return self.transform(x), y
        
        elif y is None:
            return self.transform(x)
        
        return self.transform(x), y
