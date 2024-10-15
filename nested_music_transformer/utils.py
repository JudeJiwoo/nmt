from omegaconf import DictConfig
from pathlib import Path

import torch

from . import model_zoo
from .encodec.data_utils import EncodecDataset
from .symbolic_encoding import data_utils

def wandb_style_config_to_omega_config(wandb_conf):
  # remove wandb related config
  for wandb_key in ["wandb_version", "_wandb"]:
    if wandb_key in wandb_conf:
      del wandb_conf[wandb_key] # wandb-related config should not be overrided! 

  # remove nonnecessary fields such as desc and value
  for key in wandb_conf:
    if 'desc' in wandb_conf[key]:
      del wandb_conf[key]['desc']
    if 'value' in wandb_conf[key]:
      wandb_conf[key] = wandb_conf[key]['value']
  return wandb_conf

def get_dir_from_wandb_by_code(wandb_dir: Path, code:str) -> Path:
  for dir in wandb_dir.iterdir():
    if dir.name.endswith(code):
      return dir
  print(f'No such code in wandb_dir: {code}')
  return None

def get_best_ckpt_path_and_config(wandb_dir, code):
  dir = get_dir_from_wandb_by_code(wandb_dir, code)
  if dir is None:
    raise ValueError('No such code in wandb_dir')
  ckpt_dir = dir / 'files' / 'checkpoints'
  pt_fns = sorted(list(ckpt_dir.glob('*.pt')), key=lambda fn: int(fn.stem.split('_')[0].replace('iter', '')))
  
  last_ckpt_fn = pt_fns[-1]
  config_path = dir / 'files'  / 'config.yaml'
  vocab_path = next(ckpt_dir.glob('vocab*'))
  metadata_path = next(ckpt_dir.glob('*metadata.json'))

  return last_ckpt_fn, config_path, metadata_path, vocab_path

def prepare_model_and_dataset_from_config(config: DictConfig, metadata_path:str, vocab_path:str):
  nn_params = config.nn_params
  dataset_name = config.dataset
  vocab_path = Path(vocab_path)
  if 'Encodec' in dataset_name:
    encodec_tokens_path = Path(f"dataset/maestro-v3.0.0-encodec_tokens")
    encodec_dataset = EncodecDataset(config, encodec_tokens_path, None, None)
    vocab_sizes = encodec_dataset.vocab.get_vocab_size()
    train_set, valid_set, test_set = encodec_dataset.split_train_valid_test_set()
    
    lm_model:model_zoo.LanguageModelTransformer= getattr(model_zoo, nn_params.model_name)(config, vocab_sizes)
  else:
    # midi_path = Path(f'dataset/MIDI_dataset/{dataset_name}')
    encoding_scheme = config.data_params.encoding_scheme
    symbolic_dataset = getattr(data_utils, dataset_name)(
                              in_vocab_file_path=vocab_path,
                              out_vocab_path=None,
                              encoding_scheme=encoding_scheme,
                              num_features=config.data_params.num_features,
                              debug=config.general.debug,
                              aug_type=config.data_params.aug_type,
                              input_length=config.train_params.input_length,
                              first_pred_feature=config.data_params.first_pred_feature,
                              )
    vocab_sizes = symbolic_dataset.vocab.get_vocab_size()
    print(f"---{nn_params.main_decoder}--- is used")
    print(f"---{dataset_name}--- is used")
    print(f"---{encoding_scheme}--- is used")
    split_ratio = config.data_params.split_ratio
    train_set, valid_set, test_set = symbolic_dataset.split_train_valid_test_set(dataset_name=config.dataset, ratio=split_ratio, seed=42, save_dir=None)

    ds_transformer = getattr(model_zoo, nn_params.model_name)(
                            vocab=symbolic_dataset.vocab,
                            input_length=config.train_params.input_length,
                            prediction_order=nn_params.prediction_order,
                            input_embedder_name=nn_params.input_embedder_name,
                            main_decoder_name=nn_params.main_decoder_name,
                            sub_decoder_name=nn_params.sub_decoder_name,
                            sub_decoder_depth=nn_params.sub_decoder.num_layer if hasattr(nn_params, 'sub_decoder') else 0,
                            sub_decoder_enricher_use=nn_params.sub_decoder.feature_enricher_use \
                              if hasattr(nn_params, 'sub_decoder') and hasattr(nn_params.sub_decoder, 'feature_enricher_use') else False,
                            dim=nn_params.main_decoder.dim_model,
                            heads=nn_params.main_decoder.num_head,
                            depth=nn_params.main_decoder.num_layer,
                            dropout=nn_params.main_decoder.dropout,
                            )
  return ds_transformer, test_set, symbolic_dataset.vocab

def add_conti_in_valid(tensor, encoding_scheme):
  new_target = tensor.clone()
  # Assuming tensor shape is [batch, sequence, features]
  # Create a shifted version of the tensor
  shifted_tensor = torch.roll(new_target, shifts=1, dims=1)
  # The first element of each sequence cannot be a duplicate by definition
  shifted_tensor[:, 0, :] = new_target[:, 0, :] + 1
  
  # Identify where the original and shifted tensors are the same (duplicates)
  duplicates = new_target == shifted_tensor
  # TODO: convert hard-coded part
  # convert values into False except the 1st and 2nd features
  if encoding_scheme == 'nb':
    if tensor.shape[2] == 5:
      # change beat, instrument
      duplicates[:, :, 0] = False
      duplicates[:, :, 3] = False
      duplicates[:, :, 4] = False
    elif tensor.shape[2] == 4:
      # change beat
      duplicates[:, :, 0] = False
      duplicates[:, :, 2] = False
      duplicates[:, :, 3] = False
    elif tensor.shape[2] == 7:
      # change beat, chord, tempo
      duplicates[:, :, 0] = False
      duplicates[:, :, 4] = False
      duplicates[:, :, 5] = False
      duplicates[:, :, 6] = False
  elif encoding_scheme == 'cp':
    if tensor.shape[2] == 5:
      # change instrument
      duplicates[:, :, 0] = False
      duplicates[:, :, 1] = False
      duplicates[:, :, 3] = False
      duplicates[:, :, 4] = False
    elif tensor.shape[2] == 7:
      # change chord, tempo
      duplicates[:, :, 0] = False
      duplicates[:, :, 1] = False
      duplicates[:, :, 4] = False
      duplicates[:, :, 5] = False
      duplicates[:, :, 6] = False
  
  # Replace duplicates with 9999
  new_target[duplicates] = 9999
  return new_target

def add_conti_for_single_feature(tensor):
  new_target = tensor.clone()
  # Assuming tensor shape is [batch, sequence, features]
  # Create a shifted version of the tensor
  shifted_tensor = torch.roll(new_target, shifts=1, dims=1)
  # The first element of each sequence cannot be a duplicate by definition
  shifted_tensor[:, 0] = new_target[:, 0] + 1
  
  # Identify where the original and shifted tensors are the same (duplicates)
  duplicates = new_target == shifted_tensor
  # Replace duplicates with 9999
  new_target[duplicates] = 9999
  return new_target