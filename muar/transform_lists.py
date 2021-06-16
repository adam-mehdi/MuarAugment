import albumentations as A
import kornia.augmentation as K

################################################ albumentations_list ###############################

def albumentations_list(MAGN: int = 4):
    """
    Returns standard list of albumentations transforms, each of mangitude `MAGN`.
    
    Args:
        MAGN (int): Magnitude of each transform in the returned list.
    """
    M = MAGN
    transform_list = [
        # PIXEL-LEVEL
        A.RandomContrast(limit=M*.1, always_apply=True),
        A.RandomBrightness(limit=M*.1, always_apply=True), 
        A.Equalize(always_apply=True),
        A.OpticalDistortion(distort_limit=M*.2, shift_limit=M*.1, always_apply=True),
        A.RGBShift(r_shift_limit=M*10, g_shift_limit=M*10, b_shift_limit=M*10, always_apply=True),
        A.ISONoise(color_shift=(M*.01, M*.1),intensity=(M*.02, M*.2), always_apply=True),
        A.RandomFog(fog_coef_lower=M*.01, fog_coef_upper=M*.1, always_apply=True),
        A.CoarseDropout(max_holes=M*10, always_apply=True),
        A.GaussNoise(var_limit=(M,M*50), always_apply=True),

        # SPATIAL
        A.Rotate(always_apply=True),
        A.Transpose(always_apply=True),
        A.NoOp(always_apply=True),
        A.ElasticTransform(alpha=M*.25, sigma=M*3, alpha_affine=M*3, always_apply=True),
        A.GridDistortion(distort_limit=M*.075, always_apply=True)
    ]
    return transform_list
   
    
######################################################## kornia_list #############################

def kornia_list(MAGN: int = 4):
    """
    Returns standard list of kornia transforms, each with magnitude `MAGN`.
    
    Args:
        MAGN (int): Magnitude of each transform in the returned list.
    """
    transform_list = [
                # SPATIAL
                K.RandomHorizontalFlip(p=1),
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

