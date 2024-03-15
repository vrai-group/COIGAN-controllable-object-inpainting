import os
import logging
from omegaconf import OmegaConf, DictConfig

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from COIGAN.training.data.datasets_loaders.shape_dataloader import ShapeObjectDataloader
from COIGAN.training.data.datasets_loaders.object_dataloader import ObjectDataloader

from COIGAN.training.data.augmentation.augmentor import Augmentor
from COIGAN.training.data.augmentation.augmentation_presets import augmentation_presets_dict
from COIGAN.training.data.datasets_loaders.jsonl_object_dataset import JsonLineObjectDataset, JsonLineMaskObjectDataset
from COIGAN.training.data.datasets_loaders.coigan_severstal_steel_defects_dataset import CoiganSeverstalSteelDefectsDataset

from COIGAN.training.data.datasets_loaders.jsonl_dataset import JsonLineDatasetBase, JsonLineDataset
from COIGAN.training.data.datasets_loaders.jsonl_segm_dataset import JsonLineDatasetSegm
from COIGAN.training.data.datasets_loaders.severstal_steel_defect import SeverstalSteelDefectDataset
from COIGAN.training.data.datasets_loaders.concrete_crack_conglomerate_dataset import ConcreteCrackConglomerateDataset

from COIGAN.utils.ddp_utils import data_sampler

LOGGER = logging.getLogger(__name__)


def make_dataloader(config: OmegaConf, rank=None, validation=False):
    """
    Method to create the dataset by params in config

    Args:
        config (OmegaConf): the config object with the data params used to build the dataset
        rank (int): the rank of the process (if passed is used to set the seed of the dataset)
        val (bool): if True the dataloader is created for validation
    """
    rank = rank if rank is not None else 0 # if no rank is passed, set it to 0
    seed = config.data.seed * (rank + 1) # multiply the main seed by the rank to get a different seed for each process

    # create the dataset
    if config.data.kind == "severstal-steel-defect":
        dataset = make_severstal_steel_defect(config.data, seed=seed)
    
    elif config.data.kind == "segmentation_jsonl":
        dataset = make_segmentation_jsonl(config.data, validation=validation, seed=seed)

    else:
        raise "Dataset kind not supported"

    # extracting the worker_init_fn from the dataset if it exists
    worker_init_fn = dataset.on_worker_init if hasattr(dataset, "on_worker_init") else None

    # encapsulate the dataset in the torch dataloader
    dataloader = DataLoader(
        dataset=dataset,
        sampler=data_sampler(
            dataset, 
            shuffle=config.data.dataloader_shuffle,
            distributed=config.distributed
        ),
        worker_init_fn=worker_init_fn,
        **config.data.torch_dataloader_kwargs
    )
    
    return dataloader


def make_severstal_steel_defect(config: DictConfig, seed: int = None):
    """
    Method to preparare the dataset object 
    for the severstal steel defect dataset.

    Args:
        config (OmegaConf): the data config object
    """

    # load the base dataset
    base_dataset = ImageFolder(
        **config.base_dataset_kwargs,
        transform=augmentation_presets_dict["base_imgs_preset"]
    )

    # load the reference dataset (if the use_ref_disc flag is set)
    ref_dataset = ImageFolder(
        **config.ref_dataset_kwargs,
        transform=augmentation_presets_dict["base_imgs_preset"]
    ) if config.use_ref_disc else None

    # create the augmentor object
    augmentor = Augmentor(
        transforms=augmentation_presets_dict[config.augmentation_sets.mask_aug],
        only_imgs_transforms=augmentation_presets_dict[config.augmentation_sets.img_aug]
    ) if config.augmentation_sets is not None else None

    # generate the paths for the object datasets
    object_datasets_paths = [
        os.path.join(config.object_datasets.base_path, object_dataset_name)
        for object_dataset_name in config.object_datasets.names
    ]

    # create the shape dataset
    shape_dataloaders = [
        ShapeObjectDataloader(
            JsonLineMaskObjectDataset(
                object_dataset_path,
                binary=config.object_datasets.binary,
                augmentor=augmentor
            ),
            seed=seed,
            **config.shape_dataloader_kwargs
        )
        for object_dataset_path in object_datasets_paths
    ] if config.different_shapes else None

    # create the object dataset
    object_dataloaders = [
        ObjectDataloader(
            JsonLineObjectDataset(
                object_dataset_path,
                binary=config.object_datasets.binary,
                augmentor=augmentor
            ),
            seed=seed,
            **config.object_dataloader_kwargs
        )
        for object_dataset_path in object_datasets_paths
    ]

    # create the COIGAN dataloader
    dataset = CoiganSeverstalSteelDefectsDataset(
        base_dataset,
        config.object_datasets.classes,
        object_dataloaders,
        shape_dataloaders=shape_dataloaders,
        ref_dataset=ref_dataset,
        seed=seed,
        **config.coigan_dataset_kwargs
    )

    return dataset


def make_segmentation_jsonl(config: DictConfig, seed: int = None, validation: bool = False):
    """
    Method to preparare the dataset object 
    for the severstal steel defect dataset.

    Args:
        config (OmegaConf): the data config object
    """

    # create the augmentor object
    augmentor = Augmentor(
        transforms=augmentation_presets_dict[config.augmentation_sets.mask_aug],
        only_imgs_transforms=augmentation_presets_dict[config.augmentation_sets.img_aug]
    ) if not validation else None

    if not validation:
        image_folder_path = config.image_folder_path
        metadata_file_path = config.metadata_file_path
        index_file_path = config.index_file_path
    else:
        image_folder_path = config.val_image_folder_path
        metadata_file_path = config.val_metadata_file_path
        index_file_path = config.val_index_file_path

    # create the dataset
    dataset = JsonLineDatasetSegm(
        image_folder_path=image_folder_path,
        metadata_file_path=metadata_file_path,
        index_file_path=index_file_path,
        classes=config.classes,
        background_class=config.background_class,
        collapse_classes=config.collapse_classes,
        augmentor=augmentor,
        masks_fields=config.masks_fields,
        binary=config.binary
    )

    return dataset

#####################
# Debugging section #
if __name__ == "__main__":
    import hydra
    from tqdm import tqdm
    from COIGAN.utils.common_utils import sample_data
    from COIGAN.utils.debug_utils import check_nan

    def check_sample(COIGAN_dataloader: CoiganSeverstalSteelDefectsDataset):
        for sample in tqdm(COIGAN_dataloader):
            base_image =        check_nan(sample["base"]) # [base_r, base_g, base_b] the original image without any masking
            ref_image =         check_nan(sample["ref"]) # [ref_r, ref_g, ref_b] the reference image used in the discriminator
            gen_in =            check_nan(sample["gen_input"]) # [base_r, base_g, base_b, mask_0, mask_1, mask_2, mask_3]
            gen_in_orig_masks = check_nan(sample["orig_gen_input_masks"]) # [mask_0, mask_1, mask_2, mask_3] the original masks without the noise
            disc_in_true =      check_nan(sample["disc_input"])# [defect_0_r, defect_0_g, defect_0_b, defect_1_r, defect_1_g, defect_1_b, defect_2_r, defect_2_g, defect_2_b, defect_3_r, defect_3_g, defect_3_b]    
            union_mask =        check_nan(sample["gen_input_union_mask"]) # [union_mask] the union mask of all the masks used in the generator input

    #config_path = "/home/ubuntu/hdd/COIGAN-controllable-object-inpainting/configs/training/"
    config_path = "/home/max/thesis/COIGAN-controllable-object-inpainting/configs/training/"
    @hydra.main(config_path=config_path, config_name="test_train.yaml")
    def main_debug(cfg: OmegaConf):
        dataloader = make_dataloader(cfg)
        loader = sample_data(dataloader)
        check_sample(loader)


    main_debug()





    