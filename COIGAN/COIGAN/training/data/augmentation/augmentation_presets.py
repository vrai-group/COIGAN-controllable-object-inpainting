from COIGAN.training.data.augmentation.custom_transformations import *
from torchvision.transforms.transforms import *

"""PRESETS OF TRANSFORMATIONS APPLICABLE TO IMAGES AND MASKS"""
mask_preset_1 = dChoice(
    [
        dCropUpscale((254, 512), 700),
        dRotations(180),
        dAffine(degrees=0.0, translate=0.3, scale=(0.8, 1.2), shear=25.0),
    ],
    [0, 3],
)

"""PRESEET OF TRANSFORMATIONS APPLICABLE ONLY TO IMAGES"""
imgs_preset_1 = sChoice(
    [
        ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        RandomErasing(p=0.3, scale=(0.02, 0.13), ratio=(0.3, 3.3), value="random"),
        GaussianBlur(kernel_size=5),
        RandomPosterize(p=0.3, bits=4),
        RandomAutocontrast(p=0.3),
        RandomSolarize(p=0.3, threshold=128),
        RandomAdjustSharpness(p=0.3, sharpness_factor=2.0),
        # RandomEqualize(p=0.3),
    ],
    [0, 4],
)

######################################################################
### Interactor augmentation presets

"""
PRESETS OF TRANSFORMATIONS APPLICABLE TO IMAGES AND MASKS
Use this augmentation where the masks contain the background class as the last one,
in this case rotations will generate a 1's border around the mask when it is rotated.
"""
mask_defects_bg_preset = dChoice(
    [
        dVerticalFlip(p=0.5),
        dHorizontalFlip(p=0.5),
        dRotations(35, bg_fill=1),
        #dAffine(degrees=0.0, translate=0.1, scale=(0.9, 1.1), shear=5.0),
    ],
    [1, 2],
)


"""PRESETS OF TRANSFORMATIONS APPLICABLE TO IMAGES AND MASKS"""
mask_defects_preset = dChoice(
    [
        dVerticalFlip(p=0.5),
        dHorizontalFlip(p=0.5),
        dRotations(25),
        #dAffine(degrees=0.0, translate=0.1, scale=(0.9, 1.1), shear=5.0),
    ],
    [0, 2],
)

"""PRESEET OF TRANSFORMATIONS APPLICABLE ONLY TO IMAGES"""
imgs_defects_preset = sChoice(
    [
        #ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        #RandomErasing(p=0.3, scale=(0.02, 0.13), ratio=(0.3, 3.3), value="random"),
        #GaussianBlur(kernel_size=5),
        RandomPosterize(p=0.3, bits=4),
        RandomAutocontrast(p=0.3),
        RandomSolarize(p=0.3, threshold=128),
        RandomAdjustSharpness(p=0.3, sharpness_factor=2.0),
        #RandomEqualize(p=0.3),
    ],
    [0, 3],
)

imgs_defects_preset_noop = sNoOp()

######################################################################
### Inpainting augmentation presets

"""PRESETS OF TRANSFORMATIONS APPLICABLE TO IMAGES AND MASKS"""
mask_inpainting_preset = dChoice(
    [
        #dVerticalFlip(p=0.5),
        #dHorizontalFlip(p=0.5),
        dRotations(180, expand=True),
        dAffine(degrees=0.0, translate=0.1, scale=(0.7, 1.2), shear=3.0),
    ],
    [0, 2],
)

"""PRESEET OF TRANSFORMATIONS APPLICABLE ONLY TO IMAGES"""
imgs_inpainting_preset = sNoOp()

######################################################################
### base gan augmentation presets

base_imgs_preset = Compose([
    RandomHorizontalFlip(p=0.5),
    RandomHorizontalFlip(p=0.5),
    ToTensor()
])

######################################################################
### Dict mapping all the presets

augmentation_presets_dict = {
    "mask_preset_1": mask_preset_1,
    "imgs_preset_1": imgs_preset_1,
    "mask_defects_preset": mask_defects_preset,
    "imgs_defects_preset": imgs_defects_preset,
    "mask_defects_bg_preset": mask_defects_bg_preset,
    "imgs_defects_preset_noop": imgs_defects_preset_noop,
    "mask_inpainting_preset": mask_inpainting_preset,
    "imgs_inpainting_preset": imgs_inpainting_preset,
    "base_imgs_preset": base_imgs_preset,
}


######################################################################
### DEBUG
if __name__ == "__main__":

    # Test the augmentation presets
    from PIL import Image
    import matplotlib.pyplot as plt
    import numpy as np

    # Load the image and the mask
    img = Image.open("")
    mask = Image.open("")

    while True:
        # Apply the augmentation presets
        img_aug, mask_aug = mask_defects_preset(img, mask)
        img_aug = imgs_defects_preset(img_aug)

        # Show the results
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(np.asarray(img_aug))
        ax[1].imshow(np.asarray(mask_aug))
        plt.show()