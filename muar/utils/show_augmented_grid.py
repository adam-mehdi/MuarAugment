import numpy as np
from typing import Union
import kornia
import kornia.augmentation as K
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as TF
import muar

def show_augmented_grid(image: Union[np.ndarray, torch.Tensor], 
                        transform: Union[K.augmentation, muar.BatchRandAugment] 
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
