import albumentations as A

def albumentations_list(MAGN):
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
