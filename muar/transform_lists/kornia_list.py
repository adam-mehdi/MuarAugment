import kornia.augmentation as K

def kornia_list(MAGN):
    transform_list = [
                # SPATIAL
                K.RandomHorizontalFlip(p=1),
                K.RandomVerticalFlip(p=1),
                K.RandomRotation(degrees=90., p=1),
                K.RandomAffine(degrees=MAGN*5.,shear=MAGN/5,translate=MAGN/20, p=1),
                K.RandomPerspective(distortion_scale=MAGN/25, p=1),

                # PIXEL-LEVEL
                K.ColorJitter(brightness=MAGN/30, p=1),                      # brightness
                K.ColorJitter(saturation=MAGN/30, p=1),                      # saturation
                K.ColorJitter(contrast=MAGN/30, p=1),                        # contrast
                K.ColorJitter(hue=MAGN/30, p=1),                             # hue
                K.ColorJitter(p=0),                                          # identity
                K.RandomMotionBlur(kernel_size=2*(MAGN//3)+1, angle=MAGN, direction=1., p=1),
                K.RandomErasing(scale=(MAGN/100,MAGN/50), ratio=(MAGN/20,MAGN), p=1),
            ]
    return transform_list