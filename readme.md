# Nested Music Transformer

This repository contains the official implementation of **Nested Music Transformer**, as described in the paper:

**Nested Music Transformer**  
Jiwoo Ryu, Hao-Wen Dong, Jongmin Jung, and Dasaem Jeong  
_The International Society for Music Information Retrieval (ISMIR)_, 2024  
[[Web Demo](https://judejiwoo.github.io/nested-music-transformer-demo/)]  
[[Paper](https://arxiv.org/abs/2408.01180)]

---

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Preprocessing](#preprocessing)
- [Training](#training)
- [Generation](#generation)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

---

## Introduction

The **Nested Music Transformer** is a model designed to enhance sequence generation tasks for music composition. It introduces a novel nested attention mechanism that enables better hierarchical structure modeling within musical sequences. This approach is particularly useful for capturing both local and global patterns in music.

---

## Installation

To install the necessary dependencies, please follow these steps:

1. **Clone this repository:**
    ```bash
    git clone https://github.com/judejiwoo/nested-music-transformer.git
    cd nested-music-transformer
    ```

2. **Set up a pipenv environment and install dependencies:**
    ```bash
    pipenv install --python 3.x  # Replace 3.x with your specific Python version, e.g., 3.8
    ```

3. **Activate the pipenv shell:**
    ```bash
    pipenv shell
    ```

4. **Run the project:**
    After activating the environment, you can run your project scripts as usual:
    ```bash
    python your_script.py  # Replace 'your_script.py' with the main script name
    ```

This will create an isolated environment with the exact dependencies listed in `requirements.txt` using `pipenv`.

---

## Preprocessing

You can download and preprocess the dataset following the instructions in the data_representation folder.

## Training

### Pretrained Models

The pretrained models can be found [here](https://).

### Training Guide

This project uses Hydra for flexible configuration management, allowing users to specify training settings via YAML files. 
Also the training sessions are designed to use wandb so please set correct project and entitiy name in the train.py script.

    ```
    wandb.init(
      project="Nested_Music_Transformer",
      entity="judejiwoo",
      name=experiment_name,
      config = OmegaConf.to_container(config)
    )
    ```

To begin training, follow these steps:

1. **Configuration Files**:
   - The primary configuration files are located in `nested_music_transformer/symbolic_yamls/` or `encoding_yamls`, including:
     - **`config.yaml`**: Contains general settings such as device usage, logging options, and training parameters.
     - **`nn_params.yaml`**: Specifies model parameters, including the encoding scheme (e.g., `remi`), number of features, and architecture details. It is crucial to set the correct `nn_params` to match the desired encoding scheme and dataset requirements according to your purpose.

2. **Main Training Script**:
   - The main training script, `train.py`, is located at:
     ```
     /home/jude/userdata/symbolic-music-encoding/nmt/train.py
     ```
   - This script automatically loads the configuration files and supports both single and multi-GPU training using Distributed Data Parallel (DDP).

3. **Running the Training Script**:
   - To launch a training run, specify any necessary parameters directly from the command line like follows:
     ```bash
     CUDA_VISIBLE_DEVICES=1 python3 train.py use_fp16=True nn_params=remi5 dataset=SOD train_params.batch_size=8 train_params.input_length=7168 nn_params.main_decoder.num_layer=10 nn_params.model_dropout=0.1 train_params.decay_step_rate=0.9 general.make_log=False
     ```
   - **Key Parameters**:
     - `nn_params`: Choose the correct encoding scheme and feature settings (e.g., `remi5` for REMI encoding with five features).
     - `dataset`: Specify the dataset name (e.g., `SOD`).
     - `train_params`: Adjust other training settings, like batch size and input length.
     - `make_log`: Decide whether to log this experiment in your wandb project.
     - `first_pred_feature`: This parameter is used for compound shifting. Note: This works for NB encoding only. Unlike in the paper, the sub-token name "metric" in NB is set as a "type" for convenience when comparing encoding schemes. If you choose "type" as the `first_pred_feature`, compound shifting will not be applied. We recommend selecting "pitch" as the first option to achieve better NLL results.

