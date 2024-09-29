import os
import copy
from pathlib import Path
from datetime import datetime

import torch
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group

import wandb
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from double_sequential.symbolic_encoding import data_utils, decoding_utils
from double_sequential.symbolic_encoding.data_utils import get_emb_total_size
from double_sequential import model_zoo, trainer
from double_sequential.train_utils import SingleClassNLLLoss, MultiClassNLLLoss, CosineAnnealingWarmUpRestarts, EncodecFlattenLoss, EncodecMultiClassLoss, CosineLRScheduler
from double_sequential.encodec.data_utils import EncodecDataset
from run_evaluation import main as run_evaluation

def ddp_setup(rank, world_size, backend='nccl'):
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '12355'
  init_process_group(backend, rank=rank, world_size=world_size)
  torch.cuda.set_device(rank)

def generate_experiment_name(config):
  # Access the nn_params name from the hydra config
  # hydra_cfg = HydraConfig.get()

  # add base hyperparameters to the experiment name
  # nn_params_name = hydra_cfg.runtime.choices.nn_params
  dataset_name = config.dataset
  encoding_name = config.data_params.encoding_scheme
  num_features = config.data_params.num_features
  input_embedder_name = config.nn_params.input_embedder_name
  sub_decoder_name = config.nn_params.sub_decoder_name
  batch_size = config.train_params.batch_size
  num_layers = config.nn_params.main_decoder.num_layer
  input_length = config.train_params.input_length
  first_pred_feature = config.data_params.first_pred_feature

  # Add target hyperparameters to the experiment name
  # dropout
  main_dropout = config.nn_params.main_decoder.dropout 
  # learning rate
  lr_decay_rate = config.train_params.decay_step_rate

  # Combine the information into a single string for the experiment name
  experiment_name = f"{dataset_name}_{encoding_name}{num_features}_{input_embedder_name}_{sub_decoder_name}_firstpred:{first_pred_feature}_inputlen{input_length}_nlayer{num_layers}_batch{batch_size}\
  _dropout{main_dropout}_lrdecay{lr_decay_rate}"
  return experiment_name

def setup_log(config):
  if config.general.make_log:
    experiment_name = generate_experiment_name(config)
    wandb.init(
      project="SymbolicEncoding_8th",
      entity="clayryu",
      name=experiment_name,
      config = OmegaConf.to_container(config)
    )
    save_dir = wandb.run.dir + '/checkpoints/'
    Path(save_dir).mkdir(exist_ok=True, parents=True)
  else:
    now = datetime.now()
    save_dir = 'wandb/debug/checkpoints/'+now.strftime('%y-%m-%d')
    Path(save_dir).mkdir(exist_ok=True, parents=True)
  return save_dir

def preapre_sybmolic(config, save_dir, rank) -> trainer.LanguageModelTrainer:
  nn_params = config.nn_params
  dataset_name = config.dataset
  encoding_scheme = config.data_params.encoding_scheme
  num_features = config.data_params.num_features

  vocab_dir = Path(f'vocab/vocab_{dataset_name}')
  in_vocab_file_path = vocab_dir / f'vocab_{dataset_name}_{encoding_scheme}{num_features}.json'
  out_vocab_path = Path(save_dir) / f'vocab_{dataset_name}_{encoding_scheme}{num_features}.json'

  symbolic_dataset = getattr(data_utils, dataset_name)(
                            in_vocab_file_path=in_vocab_file_path,
                            out_vocab_path=out_vocab_path,
                            encoding_scheme=encoding_scheme,
                            num_features=num_features,
                            debug=config.general.debug,
                            aug_type=config.data_params.aug_type,
                            input_length=config.train_params.input_length,
                            first_pred_feature=config.data_params.first_pred_feature,
                            )

  config = get_emb_total_size(config, symbolic_dataset.vocab)
  print(f"---{nn_params.main_decoder_name}--- is used")
  print(f"---{dataset_name}--- is used")
  print(f"---{encoding_scheme}--- is used")
  split_ratio = config.data_params.split_ratio
  trainset, validset, testset = symbolic_dataset.split_train_valid_test_set(dataset_name=config.dataset, ratio=split_ratio, seed=42, save_dir=save_dir)

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
  
  total_params = sum(p.numel() for p in ds_transformer.parameters())
  print(f"Total number of parameters is: {total_params}")
  # log in wandb
  if config.general.make_log:
    wandb.log({'lm_total_params': total_params}, step=0)
  
  focal_alpha = config.train_params.focal_alpha
  focal_gamma = config.train_params.focal_gamma
  if encoding_scheme == 'remi':
    loss_fn = SingleClassNLLLoss(focal_alpha=focal_alpha, focal_gamma=focal_gamma)
  elif encoding_scheme in ['cp', 'nb']:
    loss_fn = MultiClassNLLLoss(feature_list=symbolic_dataset.vocab.feature_list, focal_alpha=focal_alpha, focal_gamma=focal_gamma)

  optimizer = torch.optim.AdamW(ds_transformer.parameters(), lr=config.train_params.initial_lr, betas=(0.9, 0.95), eps=1e-08, weight_decay=0.01)
  scheduler_dict = {'not-using': None,'cosineannealingwarmuprestarts': CosineAnnealingWarmUpRestarts, 'cosinelr': CosineLRScheduler}
  if scheduler_dict[config.train_params.scheduler] == CosineAnnealingWarmUpRestarts:
    scheduler = scheduler_dict[config.train_params.scheduler](optimizer, T_0=config.train_params.num_steps_per_cycle, T_mult=2, eta_min=0, eta_max=config.train_params.max_lr,  T_up=config.train_params.warmup_steps , gamma=config.train_params.gamma)
  elif scheduler_dict[config.train_params.scheduler] == CosineLRScheduler:
    scheduler = scheduler_dict[config.train_params.scheduler](optimizer, total_steps=config.train_params.num_iter * config.train_params.decay_step_rate, warmup_steps=config.train_params.warmup_steps, lr_min_ratio=0.1, cycle_length=1.0)
  else:
    scheduler = None

  # get resolution
  in_beat_resolution_dict = {'BachChorale': 4, 'Pop1k7': 4, 'Pop909': 4, 'SOD': 12, 'LakhClean': 4, 'SymphonyMIDI': 8}
  in_beat_resolution = in_beat_resolution_dict[dataset_name]
  midi_decoder_dict = {'remi':'MidiDecoder4REMI', 'cp':'MidiDecoder4CP', 'nb':'MidiDecoder4NB'}
  midi_decoder = getattr(decoding_utils, midi_decoder_dict[encoding_scheme])(vocab=symbolic_dataset.vocab, in_beat_resolution=in_beat_resolution, dataset_name=dataset_name)

  trainer_option_dict = {'remi': 'SingleClassLMTrainer', 'cp': 'MultiClassLMTrainer', 'nb':'MultiClassLMTrainer'}
  trainer_option = trainer_option_dict[encoding_scheme]
  sampling_method = 'top_p'
  sampling_threshold = 0.9
  sampling_temperature = 2.0

  training_module = getattr(trainer, trainer_option)(
                            model=ds_transformer,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            loss_fn=loss_fn,
                            midi_decoder=midi_decoder,
                            train_set=trainset,
                            valid_set=validset,
                            save_dir=save_dir,
                            vocab=symbolic_dataset.vocab,
                            use_ddp=config.use_ddp,
                            use_fp16=config.use_fp16,
                            world_size=config.train_params.world_size,
                            batch_size=config.train_params.batch_size,
                            infer_target_len=symbolic_dataset.mean_len_tunes,
                            gpu_id=rank,
                            sampling_method=sampling_method,
                            sampling_threshold=sampling_threshold,
                            sampling_temperature=sampling_temperature,
                            config=config
                            )
  return training_module

def prepare_encodec(config, save_dir, rank):
  save_dir = setup_log(config)
  nn_params = config.nn_params
  encoding_scheme = config.data_params.encoding_scheme

  vocab_dir = Path(f'vocab/vocab_MaestroEncodec')
  Path(vocab_dir).mkdir(exist_ok=True, parents=True)
  in_vocab_file_path = vocab_dir / f'maestro-v3.0.0-in_vocab.json'
  out_vocab_path = Path(save_dir) / f'maestro-v3.0.0-in_vocab.json'

  token_path = Path(f"dataset/encodec_dataset/maestro-v3.0.0-encodec_{config.data_type}")
  encodec_dataset = EncodecDataset(
                                  in_vocab_file_path=in_vocab_file_path,
                                  out_vocab_path=out_vocab_path,
                                  encoding_scheme=encoding_scheme,
                                  input_length=config.train_params.input_length,
                                  token_path=token_path
                                  ) 
  
  trainset, validset, testset = encodec_dataset.split_train_valid_test_set()
  
  ds_transformer = getattr(model_zoo, nn_params.model_name)(
                          vocab=encodec_dataset.vocab,
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
  
  total_params = sum(p.numel() for p in ds_transformer.parameters())
  print(f"Total number of parameters is: {total_params}")
  # log in wandbtrain_dataset, test_dataset
  if config.general.make_log:
    wandb.log({'lm_total_params': total_params})

  if encoding_scheme == 'remi':
    loss_fn = EncodecFlattenLoss(feature_list=encodec_dataset.vocab.feature_list)
  elif encoding_scheme == 'nb' or encoding_scheme == 'nb_delay':
    loss_fn = EncodecMultiClassLoss(feature_list=encodec_dataset.vocab.feature_list)

  optimizer = torch.optim.AdamW(ds_transformer.parameters(), lr=config.train_params.initial_lr, betas=(0.9, 0.95), eps=1e-08, weight_decay=0.01)
  scheduler_dict = {'not-using': None,'cosineannealingwarmuprestarts': CosineAnnealingWarmUpRestarts, 'cosinelr': CosineLRScheduler}
  if scheduler_dict[config.train_params.scheduler] == CosineAnnealingWarmUpRestarts:
    scheduler = scheduler_dict[config.train_params.scheduler](optimizer, T_0=config.train_params.num_steps_per_cycle, T_mult=2, eta_min=0, eta_max=config.train_params.max_lr,  T_up=config.train_params.warmup_steps , gamma=config.train_params.gamma)
  elif scheduler_dict[config.train_params.scheduler] == CosineLRScheduler:
    scheduler = scheduler_dict[config.train_params.scheduler](optimizer, total_steps=config.train_params.num_iter * config.train_params.decay_step_rate, warmup_steps=config.train_params.warmup_steps, lr_min_ratio=0.1, cycle_length=1.0)
  else:
    scheduler = None

  trainer_option_dict = {'remi': 'EncodecFlattenTrainer', 'nb':'EncodecMultiClassTrainer', 'nb_delay':'EncodecMultiClassTrainer'}
  trainer_option = trainer_option_dict[encoding_scheme]
  infer_target_len_dict = {'remi': 6000, 'nb': 1500, 'nb_delay': 1500}
  infer_target_len = infer_target_len_dict[encoding_scheme]
  sampling_method = None
  sampling_threshold = 1.0
  sampling_temperature = 1.0

  training_module = getattr(trainer, trainer_option)(
                            model=ds_transformer,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            loss_fn=loss_fn,
                            midi_decoder=None,
                            train_set=trainset,
                            valid_set=validset,
                            save_dir=save_dir,
                            vocab=encodec_dataset.vocab,
                            use_ddp=config.use_ddp,
                            use_fp16=config.use_fp16,
                            world_size=config.train_params.world_size,
                            batch_size=config.train_params.batch_size,
                            infer_target_len=infer_target_len,
                            gpu_id=rank,
                            sampling_method=sampling_method,
                            sampling_threshold=sampling_threshold,
                            sampling_temperature=sampling_temperature,
                            config=config
                            )
  return training_module

def run_train_exp(rank, config, world_size:int=1):
  if config.use_ddp: ddp_setup(rank, world_size)
  config = copy.deepcopy(config)
  config.train_params.world_size = world_size
  if rank != 0:
    config.general.make_log = False
    config.general.infer_and_log = False

  save_dir = setup_log(config)
  if 'encodec' in config.dataset.lower():
    training_module = prepare_encodec(config, save_dir, rank)
  else:
    training_module = preapre_sybmolic(config, save_dir, rank)
  training_module.train_by_num_iter(config.train_params.num_iter)

  if not 'encodec' in config.dataset.lower():
    exp_code = [x for x in save_dir.split('/') if 'run-' in x][0]
    mean_nll = run_evaluation(exp_code)
    wandb.log({'evaluated_mean_nll': mean_nll})

@hydra.main(version_base=None, config_path="./double_sequential/symbolic_yamls/", config_name="config")
def main(config: DictConfig):
  if config.use_ddp:
    world_size = torch.cuda.device_count()
    mp.spawn(run_train_exp, args=(config, world_size), nprocs=world_size)
    destroy_process_group()
  else:
    run_train_exp(0, config) # single gpu

if __name__ == "__main__":
  main()