import os
import cv2
import logging

from tqdm import tqdm
from omegaconf import DictConfig

from COIGAN.utils.common_utils import sample_data
from COIGAN.training.data.datasets_loaders import make_dataloader
from COIGAN.inference.coigan_inference import COIGANinference

from COIGAN.training.data.datasets_loaders import JsonLineDatasetBase
from COIGAN.training.data.dataset_generators import JsonLineDatasetBaseGenerator
from COIGAN.training.data.dataset_generators.severstal_steel_defect_dataset_preprcessor import SeverstalSteelDefectPreprcessor
preprocess = SeverstalSteelDefectPreprcessor.preprocess # (image, masks, tile_size=(256, 256))

LOGGER = logging.getLogger(__name__)

class COIGANaugment:
    """
    Object aimed to augment the dataset.
    the object is responsible to take an input dataset, copy it to andother dataset,
    and add to it images generated from a base set and a set of masks.
    """

    def __init__(
            self,
            config: DictConfig
    ):
        """
        Init method of the COIGAN augment class,
        this class is used to load the model and augment the dataset.

        Args:
            config (DictConfig): config of the model
        """
        # save the config
        self.config = config

        # augmentation parameters
        self.copy_source_dataset = config.copy_source_dataset
        self.tile_size = config.tile_size
        self.batch_size = config.batch_size
        self.n_extra_samples = config.n_extra_samples
        self.input_dataset_folder = config.input_dataset_folder
        self.output_dataset_folder = config.output_dataset_folder

        if self.copy_source_dataset: self.input_dataset_image_folder = os.path.join(self.input_dataset_folder, "data")
        self.output_dataset_image_folder = os.path.join(self.output_dataset_folder, "data")
        
        # load the dataloader
        self.dataloader = sample_data(make_dataloader(config))

        # load the model
        self.model = COIGANinference(config)

        # load source dataset
        if self.copy_source_dataset:
            self.input_dataset = JsonLineDatasetBase(
                metadata_file_path = os.path.join(self.input_dataset_folder, "dataset.jsonl"),
                index_file_path = os.path.join(self.input_dataset_folder, "index"),
                binary = True
            )
            self.input_dataset.on_worker_init()

        # create the dataset generator
        self.dataset_generator = JsonLineDatasetBaseGenerator(
            self.output_dataset_folder,
            dump_every = 10000,
            binary = True
        )

        # create the output folders
        os.makedirs(self.output_dataset_folder, exist_ok=True) # create the data folder
        os.makedirs(self.output_dataset_image_folder, exist_ok=True) # create the data folder (for the images)

    
    def copy_source(self):
        """
        This method is aimed to copy the whole source dataset into the output dataset,
        leaving it open to insert the augmented images.
        """
        LOGGER.info("Copying the source dataset into the output dataset..")
        pbar = tqdm(total = len(self.input_dataset))
        for sample in self.input_dataset:
            self.dataset_generator.insert(sample)
            #copy the corresponding image
            input_image_path = os.path.join(self.input_dataset_image_folder, sample["img"])
            output_image_path = os.path.join(self.output_dataset_image_folder, sample["img"])
            os.system(f"cp {input_image_path} {output_image_path}")
            pbar.update()
        pbar.close()
        LOGGER.info("Copying the input dataset done.")


    def augment(self):
        """
        Method to augment the dataset.
        """
        # before the augmentation launch the copy source method
        if self.copy_source_dataset: self.copy_source()

        LOGGER.info("Starting the augmentation process..")
        aug_idx = 0 # index of the augmented images
        pbar = tqdm(total = self.n_extra_samples)
        while True:
            # inference on the next sample
            sample = next(self.dataloader)
            t_masks = sample["orig_gen_input_masks"]
            inpainted_img = self.model(sample["gen_input"])
            
            # convert the masks to numpy, reordering the channels
            # shape: (batch, cls, h, w) -> (batch, h, w, cls)
            batch_masks = t_masks.permute(0, 2, 3, 1).numpy().astype("uint8")

            # save the inpainted image in the target folder
            for i in range(len(inpainted_img)):
                # extracting the image and the masks from the batch
                img = inpainted_img[i]
                masks = batch_masks[i]

                # preprocess the image and the masks
                # NOTE: the preprocess method return a list of images and a list of metadata 
                images, metadata_lst = preprocess(img, masks, self.tile_size)
                for image, metadata in zip(images, metadata_lst):
                    
                    # saving the generated image
                    cv2.imwrite(os.path.join(self.output_dataset_image_folder, f"aug_{aug_idx}.png"), image)
                    metadata["img"] = f"aug_{aug_idx}.png"
                    aug_idx += 1
                    pbar.update()

                # saving the sample in the output dataset
                self.dataset_generator.insert(metadata_lst)
                
                # check if the number of augmented images is enough
                if aug_idx >= self.n_extra_samples: 
                    self.dataset_generator.close()
                    pbar.close()
                    LOGGER.info("Augmentation done.")
                    return

                
                