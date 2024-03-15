import os
import logging

os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

import hydra
from omegaconf import OmegaConf

from COIGAN.inference.coigan_augment import COIGANaugment

LOGGER = logging.getLogger(__name__)

#conf_file  = "test_augment_ssdd.yaml"
#conf_file  = "test_augment_cccd.yaml"
conf_file = "test_augment_bridge.yaml"

@hydra.main(config_path="../configs/augment/", config_name=conf_file, version_base="1.1")
def main(config: OmegaConf):
    
    #resolve the config inplace
    OmegaConf.resolve(config)
    
    LOGGER.info(f'Config: {OmegaConf.to_yaml(config)}')
    
    OmegaConf.save(config, os.path.join(os.getcwd(), 'config.yaml')) # saving the configs to config.hydra.run.dir

    augmenter = COIGANaugment(config)
    augmenter.augment()


if __name__ == "__main__":
    main()