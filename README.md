---

<div align="center">    
 
# <img src="muar-final-design-2.JPG" width="60" height="35"/>     MuarAugment  

</div>

## Description   
MuarAugment provides the easiest way to a state-of-the-art data augmentation pipeline. 

It adapts the leading pipeline search algorithms, RandAugment<sup>[1]</sup> and the model uncertainty-based augmentation scheme<sup>[2]</sup> (called MuAugment here), and modifies them to work batch-wise, on the GPU. Kornia and albumentations are used for batch-wise and item-wise transforms respectively.

If you are looking to quickly obtain the SOTA data augmentation pipelines without the conventional trial-and-error, MuarAugment is the package for you.

## How to use   
You can install `MuarAugment` via PIP:  
```python
!pip install muaraugment
```

## Example (temp: tutorials and working examples coming soon)
```python
from muar.augmentations import BatchRandAugment, MuAugment

# muar augmentations
rand_augment = BatchRandAugment(N_TFMS=3, MAGN=4)
mu_augment = MuAugment(rand_augment, N_COMPS=4, SELECTED=2)

# model
model = LitClassifier(mu_augment)

# data
train, val, test = mnist()

# train
trainer = Trainer()
trainer.fit(model, train, val)
```

### Tutorials   
- [Overview of data augmentation policy search algorithms](https://adam-mehdi23.medium.com/automatic-data-augmentation-an-overview-and-the-sota-109ffbf43a20)

### Papers Referenced
1. Cubuk, Ekin et al. "RandAugment: Practical data augmentation with no separate search," 2019, [arXiv](http://arxiv.org/abs/1909.13719).
2. Wu, Sen et al. "On the Generalization Effects of Linear Transformations in Data Augmentation," 2020, [arXiv](https://arxiv.org/abs/2005.00695).
3. Riba, Edgar et al. "Kornia: an Open Source Differentiable Computer Vision Library for PyTorch," 2019, [arXiv](https://arxiv.org/abs/1910.02190).

