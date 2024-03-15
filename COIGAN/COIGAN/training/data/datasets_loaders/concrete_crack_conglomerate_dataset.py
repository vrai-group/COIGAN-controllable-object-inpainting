import os
import csv
import cv2
import numpy as np
import pandas as pd
import logging

from typing import Union, List



LOGGER = logging.getLogger(__name__)

class ConcreteCrackConglomerateDataset(object):
    """
        This class allow to load the Conglomerate Concrete Crack Dataset from its 
        original format.
    """

    n_classes = 1 # Number of classes presents in the dataset

    def __init__(
        self,
        dataset_path: str,
    ):
        """
            Initialize the Conglomerate Concrete Crack Dataset reader

            Args:
                dataset_path (str): Path to the dataset
                
                mode (str): Mode of the dataset. Can be "all", "defected" or "none_defected"
                    - all (default): All the images
                    - defected: Only the images with defects
                    - none_defected: Only the images without defects
                
                format (str): define the output format from the getitem method.
                    - standard (default): return the image (H,W,3) and the mask (H,W,4)
                    - mask_only: return only the mask (H,W,4)
                    
                tile_size (Union[int, list[int]]): Size of the samples in the dataset, used if the format is "mask_only" or "polygons_only"
                
        """

        # Check if the dataset path exists
        if not os.path.exists(dataset_path):
            raise FileNotFoundError("The dataset path does not exist")
        self.dataset_path = dataset_path

        #folder image + masks train Concrete


        # Load the images
        self.train_images_path = os.path.join(self.dataset_path, "images")

        # load the segmentation masks 
        self.train_masks_path = os.path.join(self.dataset_path, "masks")        

        self.all_images = os.listdir(self.train_images_path) # list of all images
        self.n_images = self.all_images.__len__()


    def get_metadata(self, index: int):
            """
                Get the metadata of one sample

                Args:
                    index (int): _description_
                
                Returns:
                    img_name (str): Name of the image
                    classes (Union[list, None]): List of the classes of the defects if any, None otherwise
                    encoded_masks (Union[list, None]): List of the encoded masks of the defects if any, None otherwise
            """

            img_name = self.all_images[index]

            return img_name


    def __getitem__(self, index: int):
        """
            Get an item from the dataset

            Args:
                index (int): Index of the item to get
        
            Returns:
                image (np.array): Image
                mask (np.array): Masks of defects 
        """

        img_name = self.get_metadata(index)

        # Get the image path
        image_path = os.path.join(self.train_images_path, img_name)
        image = cv2.imread(image_path)

        # Get the mask path
        mask_path = os.path.join(self.train_masks_path, img_name)

        masks = cv2.imread(mask_path, 0)
        masks = np.expand_dims(masks, axis=-1)
        
     

        return image, masks 

    def __iter__(self):
        """
            Iterator over the dataset

            Returns:
                image (np.array): Image
                mask (Union[np.array, None]): Masks of defects if any, None otherwise
        """

        for i in range(self.n_images):
            yield self.__getitem__(i)


    def __len__(self):
        """
            Get the length of the dataset

            Returns:
                length (int): Length of the dataset
        """

        return self.n_images
    

    @staticmethod
    def dataset_analysis_report(self, out_path: str):
        """
            Analyze the dataset

            this method return a set of statistics about the dataset:
            - a report with:
                - the number of images
                - the number of images with defects
                - the number of images without defects
                - the number of defects
                - the number of defects per class
            - a set of plots:
                - class distribution istogram
                - class area distribution istogram
                - defect and non defected istogram
        """

        report_file = os.path.join(out_path, "dataset_analysis_report.txt")
        report = open(report_file, "w")

        # Get the number of images
        report.write("Number of images: {}\n".format(self.n_images))        

if __name__ == "__main__":

    # Test the dataset class
    out_path = "/home/ubuntu/hdd/COIGAN-controllable-object-inpainting/experiments_data/tests_severstal_steel_1"
    dataset_path = "/home/ubuntu/hdd/Datasets/severstal-steel-defect-detection"
    dataset = ConcreteSteelDefectDataset(dataset_path)

    print("Number of total images: {}".format(dataset.n_images))

    for i in range(10):
        img, masks = dataset[i]

        cv2.imwrite(os.path.join(out_path, f"image_{i}.png"), img)
        for j in range(dataset.n_classes):
            cv2.imwrite(os.path.join(out_path, f"mask_{i}_{j}.png"), masks[:,:,j]*255)

    print("Done")


