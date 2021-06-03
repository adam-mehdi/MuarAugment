---

<div align="center">    
 
# <img src="muar-final-design-2.JPG" width="60" height="35"/>     MuarAugment  

</div>

## Description   
MuarAugment is a package providing the easiest way to a state-of-the-art data augmentation pipeline.

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
