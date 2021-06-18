from typing import Union

import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import to_pil_image

import kornia
import kornia.augmentation as K
from kornia.enhance import denormalize


################################################### tensor2pil ################################


from torchvision.transforms.functional import to_pil_image
from kornia.enhance import denormalize

def tensor2pil(image: torch.Tensor, 
                mean: Union[tuple, torch.Tensor], 
                std: Union[tuple, torch.Tensor]):
    "Denormalizes the image and returns it as a PIL Image."
    if not isinstance(mean, torch.Tensor): mean = torch.Tensor(mean)
    if not isinstance(std, torch.Tensor): std = torch.Tensor(std)
    denormed = denormalize(image[None], mean, std)[0]
    return to_pil_image(denormed)

#################################################### show_augmented_grid ########################

def show_augmented_grid(image: Union[np.ndarray, torch.Tensor], 
                        transform,
                        rows: int = 3, 
                        cols: int = 3, 
                        figsize: tuple = (12.,12.), 
                        denorm: bool = False,
                        mean_for_denorm: Union[tuple, float] = None,
                        std_for_denorm: Union[tuple, float] = None
                       ):
    """
    Displays grid containing `image` augmented with `transform`. Denormalize
    according to `mean_for_denorm` and `std_for_denorm` after transform if 
    `denorm` == True.
    
    `transform` can be a kornia aumentation or muar BatchRandAugment.
    """
    
    if not isinstance(image, torch.Tensor): image = TF.to_tensor(image)
    xb = image[None].repeat(rows*cols, 1, 1, 1)
    out = tfm(xb)
    if denorm:
        out = K.Denormalize(mean=torch.tensor(mean_for_denorm), 
                            std=torch.tensor(std_for_denorm))(out)
    images = [image for image in out]
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=figsize)
    for idx, image in enumerate(images):
        image = TF.to_pil_image(image)
        row = idx // cols
        col = idx % cols
        axes[row, col].axis("off")
        axes[row, col].imshow(image, cmap="gray", aspect="auto")
    plt.subplots_adjust(wspace=.025, hspace=.025)
    plt.show()
