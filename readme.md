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

The **Nested Music Transformer (NMT)** is a novel architecture designed to improve efficiency and performance in symbolic music and audio token generation. Unlike conventional models that flatten sequences, NMT decodes compound tokens (bundles of musical features) in a fully sequential and memory-efficient manner. It incorporates two transformers—a main decoder for overall sequence modeling and a sub-decoder for each token's subtleties—enabling more nuanced musical representations. NMT has demonstrated strong performance in generating symbolic music and discrete audio tokens, achieving competitive results with reduced computational demands.

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
    pipenv install --python 3.x  # Replace 3.x with your specific Python version, e.g., 3.10
    ```

3. **Activate the pipenv shell:**
    ```bash
    pipenv shell
    ```

4. **Run the project:**
    After activating the environment, you can run your project scripts as usual:
    ```bash
    python3 target_script.py  # Replace 'target_script.py' with the script name that you want to run.
    ```

This will create an isolated environment with the exact dependencies listed in `requirements.txt` using `pipenv`.

---

## Preprocessing

You can download and preprocess the dataset following the instructions in the [data_representation](./data_representation) folder.

## Training

### Pretrained Models

The pretrained models can be found [here](https://drive.google.com/drive/folders/12wI1W-cTIF_hKTv6Fp7FAH9pLIRzryyn?usp=sharing).

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
     /nmt/train.py
     ```
   - This script automatically loads the configuration files and supports both single and multi-GPU training using Distributed Data Parallel (DDP).

3. **Running the Training Script**:
   - To launch a training run, specify any necessary parameters directly from the command line like follows:
     ```bash
     CUDA_VISIBLE_DEVICES=1 python3 train.py use_fp16=True nn_params=remi5 dataset=SOD train_params.batch_size=8 train_params.input_length=7168 nn_params.main_decoder.num_layer=10 nn_params.model_dropout=0.1 train_params.decay_step_rate=0.9 general.make_log=False
     ```
     or you can just change parameters directly in the config.yaml file.
   - **Key Parameters**:
     - `nn_params`: Choose the correct encoding scheme and feature settings (e.g., `remi5` for REMI encoding with five features).
     - `dataset`: Specify the dataset name (e.g., `SOD`).
     - `train_params`: Adjust other training settings, like batch size and input length.
     - `make_log`: Decide whether to log this experiment in your wandb project.
     - `first_pred_feature`: This parameter is used for compound shifting. Note: This works for NB encoding only. Unlike in the paper, the sub-token name "metric" in NB is set as a "type" for convenience when comparing encoding schemes. If you choose "type" as the `first_pred_feature`, compound shifting will not be applied. We recommend selecting "pitch" as the first option to achieve better NLL results.

## Generation

The generation process leverages the trained models stored in Weights and Biases (wandb) experiment folders. The main script for generating outputs is `generate.py`. This script allows you to specify the experiment code corresponding to the trained model you want to use for generation.

The wandb folder should be located at:
```
/nmt/wandb/SOD_nb_5features_NMT # example of the wandb experiment folder name
```

### Usage

To run the generation script, set the correct experiment folder name and add it after the generate.py command:

```bash
python generate.py -wandb_exp_dir SOD_nb_5features_NMT -choose_selected_tunes
```
- **Key Parameters**:
  - `wandb_exp_dir`: Specifies the directory of the wandb experiment folder containing the trained model. This parameter is mandatory and determines the model used for generation.
  - `generation_type`: Defines whether the generation should be conditioned (e.g., on specific input data) or unconditioned (generated freely). Defaults to `conditioned`.
  - `sampling_method`: Determines the sampling method to use during generation. Options include `top_p` (nucleus sampling) and `top_k` (truncating the probability distribution). Defaults to `top_p`.
  - `threshold`: Sets the probability threshold for the chosen sampling method. For instance, it may represent the cumulative probability for `top_p` sampling. Defaults to `0.99`.
  - `temperature`: Controls the randomness of the sampling process. Higher values produce more diverse outputs, while lower values make the model more deterministic. Defaults to `1.15`.
  - `num_samples`: Specifies the number of samples to generate per run. Defaults to `30`.
  - `num_target_measure`: Defines the number of target measures for conditioned generation. This parameter is relevant only when using `conditioned` generation. Defaults to `4`.
  - `choose_selected_tunes`: A boolean flag that, if set, ensures generation is performed using selected tunes. This is applicable only for the SOD dataset.

## Citation

Please cite the following paper if you use the code provided in this repository.

 > Jiwoo Ryu, Hao-Wen Dong, Jongmin Jung and Dasaem Jeong, "Nested Music Transformer: Sequentially Decoding Compound Tokens in Symbolic Music and Audio Generation
," _25th International Society for Music Information Retrieval Conference (ISMIR)_, 2024.

```bibtex
@inproceedings{ryu2024nmt,
    author = {Jiwoo Ryu, Hao-Wen Dong, Jongmin Jung and Dasaem Jeong},
    title = {Nested Music Transformer},
    booktitle = {25th International Society for Music Information Retrieval Conference (ISMIR)},
    year = 2024,
}
```

## Acknowledgments
This research was supported by the National R&D Program through the National Research Foundation of Korea (NRF) funded by the Korean Government (MSIT) (RS2023-00252944, Korean Traditional Gagok Generation Using Deep Learning).
