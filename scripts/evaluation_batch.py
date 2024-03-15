#!/usr/bin/env python3

import os
import logging
import json

os.environ["HYDRA_FULL_ERROR"] = "1"

import cv2
import torch

import hydra

from tqdm import tqdm
from omegaconf import OmegaConf

from COIGAN.utils.common_utils import sample_data
from COIGAN.training.data.datasets_loaders import make_dataloader
from COIGAN.inference.coigan_inference import COIGANinference
from COIGAN.evaluation.losses.fid.fid_score import calculate_fid_given_paths

LOGGER = logging.getLogger(__name__)


def generate_inference_dataset(
        config, 
        checkpoint_path = None, 
        out_path = None
    ):
    """
    Generate the dataset from the trained model.

    Args:
        config (DictConfig): config of the model
        checkpoint_path (str): override the checkpoint path in the config
        out_path (str): override the output path in the config
    """
    
    # create the folder for the generated images
    out_path = config.generated_imgs_path if out_path is None else out_path
    os.makedirs(out_path, exist_ok=True)
    
    n_samples = config.n_samples
    dataloader = sample_data(make_dataloader(config))
    model = COIGANinference(config, checkpoint_path)
    idx = 0
    pbar = tqdm(total=n_samples)

    while True:
        # inference on the next sample
        sample = next(dataloader)
        inpainted_img = model(sample["gen_input"])

        # save the inpainted image in the target folder
        for img in inpainted_img:
            cv2.imwrite(os.path.join(out_path, f"{idx}.png"), img)
            pbar.update()
            idx += 1

            if idx >= n_samples:
                return


@hydra.main(config_path="../configs/evaluation/", config_name="test_eval_batch_cccd.yaml", version_base="1.1")
def main(config: OmegaConf):

    #resolve the config inplace
    OmegaConf.resolve(config)

    LOGGER.info(f'Config: {OmegaConf.to_yaml(config)}')

    # save ghe config in the output folder
    OmegaConf.save(config, os.path.join(os.getcwd(), 'config.yaml')) # saving the configs to config.hydra.run.dir

    # Calcultaing the reference FID between the train and test datasets
    LOGGER.info("Calculating the reference FID...")

    # calculate the reference fid between the train and test datasets
    ref_fid = calculate_fid_given_paths(
        [config.train_imgs_path, config.test_imgs_path],
        config.inc_batch_size,
        config.device,
        config.inception_dims,
        n_imgs= 1305, # number of images in the base dataset
    )
    LOGGER.info(f"Ref FID: {ref_fid}")

    results = {
        "ref FID": ref_fid,
        "FID": {}
    }

    # create a report of the evaluation
    with open(os.path.join(os.getcwd(), "report.json"), "w") as f:
        json.dump(results, f)

    # extracting all the filenames from the checkpoints_path
    # removing all the files that are not checkpoints (<int>.pt) and ordering them by the number
    min_checkpoint = config.min_checkpoint
    checkpoints = sorted([f for f in os.listdir(config.checkpoint_path) if f.endswith(".pt")], key=lambda x: int(x.split(".")[0]))
    checkpoints = [c for c in checkpoints if int(c.split(".")[0]) >= min_checkpoint] # filter the checkpoints by the min_checkpoint
    train_stats = None

    LOGGER.info(f"Found the following checkpoints: {checkpoints}")
    for i, checkpoint in enumerate(checkpoints):

        # generate the dataset for the evaluation step with the FID metric
        LOGGER.info(f"Generating the dataset for the evaluation with the checkpoint {checkpoint} ({i+1}/{len(checkpoints)})...")
        
        out_path = os.path.join(config.generated_imgs_path, checkpoint.split(".")[0])
        
        generate_inference_dataset(
            config,
            checkpoint_path=os.path.join(config.checkpoint_path, checkpoint),
            out_path=out_path
        )

        # evaluate the generated dataset with the FID metric
        fid, train_stats = calculate_fid_given_paths(
            [config.train_imgs_path, out_path],
            config.inc_batch_size,
            config.device,
            config.inception_dims,
            n_imgs=config.n_samples,
            ret_stats=True,
            stats=train_stats
        )

        LOGGER.info(f"step {checkpoint.split('.')[0]} FID:  {fid}")

        #update evaluation results
        results["FID"][checkpoint.split(".")[0]] = fid
        with open(os.path.join(os.getcwd(), "report.json"), "w") as f:
            json.dump(results, f)

    # evalution job completed message
    LOGGER.info("Evaluation job completed.")


if __name__ == "__main__":
    main()