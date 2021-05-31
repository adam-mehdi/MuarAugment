from typing import Union

import torch
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