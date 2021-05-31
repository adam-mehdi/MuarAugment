import numpy as np
from typing import Union
import kornia
import kornia.augmentation as K
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as TF

def transform_show(image: Union[np.ndarray, torch.Tensor], 
                   transform: K.augmentation, 
                   max_rows: int = 2, 
                   max_cols: int = 2, 
                   figsize: tuple = (12.,12.), 
                   denorm: bool = False,
                   mean_for_denorm: Union[tuple, float] = None,
                   std_for_denorm: Union[tuple, float] = None):
    """
    Displays grid containing `image` augmented with `transform`.
    """
    
    if not isinstance(image, torch.Tensor): image = TF.to_tensor(image)
    xb = image[None].repeat(max_rows*max_cols, 1, 1, 1)
    out = tfm(xb)
    if denorm:
        out = K.Denormalize(mean=torch.tensor(mean_for_denorm), 
                            std=torch.tensor(std_for_denorm))(out)
    images = [image for image in out]
    fig, axes = plt.subplots(nrows=max_rows, ncols=max_cols, figsize=(16.,16.))
    for idx, image in enumerate(images):
        image = TF.to_pil_image(image)
        row = idx // max_cols
        col = idx % max_cols
        axes[row, col].axis("off")
        axes[row, col].imshow(image, cmap="gray", aspect="auto")
    plt.subplots_adjust(wspace=.025, hspace=.025)
    plt.show()