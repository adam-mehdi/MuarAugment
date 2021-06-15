import numpy as np
import albumentations as A
from muar.transform_lists import albumentations_list

class RandAugmentAlbumentations:
    """
    Item-wise RandAugment using Albumentations transforms. Use this to apply 
    RandAugment within `Dataset`.
    """
    def __init__(self,
                N_TFMS: int = 2, 
                MAGN: int = 4, 
                transform_list: list = None):
         """
        Args:
            N_TFMS (int): Number of transformation in each composition.
            MAGN (int): Magnitude of augmentation applied.
            tranform: List of K transformations to sample from.
        """
        
        if transform_list == None:
                transform_list = albumentations_list(MAGN)
            
        self.transform_list = transform_list
        self.MAGN = MAGN
        self.N_TFMS = N_TFMS
      
    
    def __call__(self):
        """
        Returns a randomly sampled list of `N_TFMS` transforms from `transform_list`
        (default list provided if `None`).
        """
        sampled_tfms = np.random.choice(self.transform_list, self.N_TFMS)
        return list(sampled_tfms)
            
