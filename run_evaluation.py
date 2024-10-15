import sys
import torch
from pathlib import Path

from omegaconf import OmegaConf

from nested_music_transformer.evaluation_utils import Evaluator, wandb_style_config_to_omega_config, prepare_model_and_dataset_from_config, get_best_ckpt_path_and_config

def main(exp_code):
  wandb_dir = Path('wandb')
  ckpt_path, config_path, metadata_path, vocab_path = get_best_ckpt_path_and_config(wandb_dir, exp_code)
  config = OmegaConf.load(config_path)
  config = wandb_style_config_to_omega_config(config)
  print(ckpt_path)

  ckpt = torch.load(ckpt_path, map_location='cpu')
  model, test_set, vocab = prepare_model_and_dataset_from_config(config, metadata_path=metadata_path, vocab_path=vocab_path)
  model.load_state_dict(ckpt['model'])
  model = model.eval()

  evaluator = Evaluator(config, model, test_set, vocab, device='cuda', batch_size=16)

  evaluator.get_perplexity()
  evaluator.save_results(wandb_dir / exp_code / f'micro_evaluated_perplexity_conti_fixed.pt')
  mean_by_class = {}
  
  for key in evaluator.vocab.feature_list:
    # skip type for calculating mean as type or metric token have different meanings across encoding schemes
    if key == 'type':
      continue
    mean_nll = sum(evaluator.loss_by_class[key]) / evaluator.count_by_class[key]
    mean_by_class[key] = mean_nll
    
  # calculate micro average
  total_mean_nll = 0
  for key in mean_by_class.keys():
    total_mean_nll += mean_by_class[key] * evaluator.count_by_class[key]
  denominator = 0
  for key in mean_by_class.keys():
    denominator += evaluator.count_by_class[key]
  total_mean_nll /= denominator
  return total_mean_nll

if __name__ == '__main__':
  exp_code = sys.argv[1]
  main(exp_code)
