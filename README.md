---

<div align="center">    
 
# <img src="muar-final-design-2.JPG" width="60" height="35"/>     MuarAugment  

</div>

## Description   
MuarAugment provides the easiest way to a state-of-the-art data augmentation pipeline. 

It adapts the leading pipeline search algorithms, RandAugment<sup>[1]</sup> and the model uncertainty-based augmentation scheme<sup>[2]</sup> (called MuAugment here), and modifies them to work batch-wise, on the GPU. Kornia<sup>[3]</sup> and albumentations are used for batch-wise and item-wise transforms respectively.

If you are looking to quickly obtain the SOTA data augmentation pipelines without the conventional trial-and-error, MuarAugment is the package for you.

## How to use   
You can install `MuarAugment` via PIP:  
```python
!pip install muaraugment
```

## Examples

Modify the training logic and train like normal.

### In PyTorch Lightning
```python
from muar.augmentations import BatchRandAugment, MuAugment

 class LitModule(pl.LightningModule):
     def __init__(self, n_tfms, magn, mean, std, n_compositions, n_selected):
        ...
        rand_augment = BatchRandAugment(n_tfms, magn, mean, std)
        self.mu_transform = MuAugment(rand_augment, n_compositions, n_selected)

    def training_step(self, batch, batch_idx):
        self.mu_transform.setup(self)
        input, target = self.mu_transform((batch['input'], batch['target']))
        ...
        
trainer = Trainer()
trainer.fit(model, datamodule)
```

### In pure PyTorch
```python
from muar.augmentations import BatchRandAugment, MuAugment

def train_fn(model):

    rand_augment = BatchRandAugment(n_tfms, magn, mean, std)
    mu_transform = MuAugment(rand_augment, n_compositions, n_selected)
    
    for epoch in range(N_EPOCHS):
        for i,batch in enumerate(train_dataloader):
            mu_transform.setup(model)
            input, target = self.mu_transform(batch)
            
train_fn(model)
```

See the colab notebook tutorials (#2) for more detail on implementing MuarAugment.

## Tutorials   
1. [Overview of data augmentation policy search algorithms](https://adam-mehdi23.medium.com/automatic-data-augmentation-an-overview-and-the-sota-109ffbf43a20) (*Medium*)
2. [MuAugment tutorial and implementation in a classification task](https://colab.research.google.com/drive/1c-Zq85kteer5FGpT3a4IujoAUzAquu1-?authuser=2#scrollTo=39OBGWKgzhbR) (*Colab Notebook*)

## Papers Referenced
1. Cubuk, Ekin et al. "RandAugment: Practical data augmentation with no separate search," 2019, [arXiv](http://arxiv.org/abs/1909.13719).
2. Wu, Sen et al. "On the Generalization Effects of Linear Transformations in Data Augmentation," 2020, [arXiv](https://arxiv.org/abs/2005.00695).
3. Riba, Edgar et al. "Kornia: an Open Source Differentiable Computer Vision Library for PyTorch," 2019, [arXiv](https://arxiv.org/abs/1910.02190).

