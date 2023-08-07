---

<div align="center">    
 
# <img src="muar-final-design-2.JPG" width="60" height="35"/>     MuarAugment  

</div>

## Description   
MuarAugment is the easiest way to a randomness-based data augmentation pipeline. Neural nets learn most from data they struggle with. MuarAugment uses the model to select the data augmentations that it has most trouble with, and uses only those most difficult data for training. This has been shown empirically. Perhaps more difficult data reduces overfitting. Perhaps it exposes parts of the data distribution which the model did not see yet. Either way, MuarAugment works.

It adapts the leading pipeline search algorithms, RandAugment<sup>[1]</sup> and the model uncertainty-based augmentation scheme<sup>[2]</sup> (called MuAugment here), and modifies them to work batch-wise, on the GPU. Kornia<sup>[3]</sup> and albumentations are used for batch-wise and item-wise transforms respectively.

If you are seeking SOTA data augmentation pipelines without laborious trial-and-error, MuarAugment is the package for you.

## How to use   
You can install `MuarAugment` via PIP:  
```python
!pip install git+https://github.com/adam-mehdi/MuarAugment.git
```

## Examples

For `MuAugment`, simply modify the training logic and train like normal.

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

See the colab notebook tutorial (#1) for more detail on implementing `MuAugment`.

### RandAugment using Albumentations

`MuarAugment` also contains a straightforward implementation of RandAugment using Albumentations:
```python
class RandAugmentDataset(Dataset):
    def __init__(self, N_TFMS=0, MAGN=0, stage='train', ...):
        ...
        if stage == 'train': 
            self.rand_augment = AlbumentationsRandAugment(N_TFMS, MAGN)
        else: self.rand_augment = None

    def __getitem__(self, idx):
        ...
        transform = get_transform(self.rand_augment, self.stage, self.size)
        augmented = transform(image=image)['image']
        ...

def get_transform(rand_augment, stage='train', size=(28,28)):
    if stage == 'train':
        resize_tfm = [A.Resize(*size)]
        rand_tfms = rand_augment() # returns a list of transforms
        tensor_tfms = [A.Normalize(), ToTensorV2()]
        return A.Compose(resize_tfm + rand_tfms + tensor_tfms)
    ...
```    

See the colab notebook tutorial (#2) for more detail on `AlbumentationsRandAugment`.

## Tutorials   
1. [MuAugment tutorial and implementation in a classification task](https://github.com/adam-mehdi/MuarAugment/blob/master/MuAugmentTutorial.ipynb) (*Colab Notebook*)
2. [RandAugment tutorial in an end-to-end pipeline](https://github.com/adam-mehdi/MuarAugment/blob/master/RandAugmentTutorial.ipynb) (*Colab Notebook*)
3. [Overview of data augmentation policy search algorithms](https://adam-mehdi23.medium.com/automatic-data-augmentation-an-overview-and-the-sota-109ffbf43a20) (*Medium*)

## Papers Referenced
1. Cubuk, Ekin et al. "RandAugment: Practical data augmentation with no separate search," 2019, [arXiv](http://arxiv.org/abs/1909.13719).
2. Wu, Sen et al. "On the Generalization Effects of Linear Transformations in Data Augmentation," 2020, [arXiv](https://arxiv.org/abs/2005.00695).
3. Riba, Edgar et al. "Kornia: an Open Source Differentiable Computer Vision Library for PyTorch," 2019, [arXiv](https://arxiv.org/abs/1910.02190).

