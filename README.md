<img align="left" width="500"  src="images/vrai-logo-b.png" style="margin-right:-230px"></br></br></br></br>
<h1> COIGAN: Controllable Object Inpainting through Generative Adversarial Network applied to Defect Synthesis for Data Augmentation</h1>
</br>

Revisited COIGAN project for the IROS 2024 conference. This project exploit the COIGAN training architecture to generate a dataset of defected images from a dataset of images from the new Morandi bridge located at Genova.

# Table of Contents
- [Table of Contents](#table-of-contents)
- [Model results](#model-results)
  - [Dataset used for the training](#dataset-used-for-the-training)
  - [Sample results](#sample-results)
  - [Fid evaluation](#fid-evaluation)
- [Build the Docker image](#build-the-docker-image)
  - [Setup the build\_args file](#setup-the-build_args-file)
  - [Build the Docker image](#build-the-docker-image-1)
- [Run the Docker image](#run-the-docker-image)
- [Run the training](#run-the-training)
- [Run the evaluation](#run-the-evaluation)
- [Run the inference](#run-the-inference)

# Model results

## Dataset used for the training


## Sample results


## Fid evaluation


# Build the Docker image

## Setup the build_args file
Before launching the build of the Docker image, you need to setup the `build_args.txt` file. This file contains the arguments used by the Dockerfile to build the image. The file is structured as follows:
```bash
--build-arg KAGGLE_USERNAME=<username>
--build-arg KAGGLE_KEY=<key>
--build-arg WANDB_API_KEY=<wandb_key>
```
where `<username>` and `<key>` are the username and the key of your Kaggle account. You can find them in the `Account` section of your Kaggle profile.
The `<wandb_key>` is the key of your Weights and Biases account. You can find it in the `Settings` section of your Weights and Biases profile.

## Build the Docker image
To build the Docker image, run the following command from the path ../COIGAN-IROS-2024/Docker:
```bash
docker build $(cat build_args.txt) -t coigan-iros-2024 .
```

# Run the Docker image
To run the Docker image, run the following command from the path ../COIGAN-IROS-2024/Docker:
```bash
docker run -it --gpus all --rm coigan-iros-2024
```
Or if you need to attach to the container only a subset of the system GPUs:
```bash
docker run -it --gpus '"device=0,1"' --rm coigan-iros-2024
```

# Run the training


# Run the evaluation


# Run the inference




