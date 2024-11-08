import sys
from pathlib import Path

import torch
import argparse
from omegaconf import OmegaConf

from nested_music_transformer.evaluation_utils import wandb_style_config_to_omega_config, prepare_model_and_dataset_from_config, get_best_ckpt_path_and_config
from nested_music_transformer.evaluation_utils import Evaluator

def get_argument_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "-wandb_exp_dir",
      required=True,
      type=str,
      help="wandb experiment directory",
  )
  parser.add_argument(
      "-generation_type",
      type=str,
      choices=('conditioned', 'unconditioned'),
      default='conditioned',
      help="generation type",
  )
  parser.add_argument(
      "-sampling_method",
      type=str,
      choices=('top_p', 'top_k'),
      default='top_p',
      help="sampling method",
  )
  parser.add_argument(
      "-threshold",
      type=float,
      default=0.99,
      help="threshold",
  )
  parser.add_argument(
      "-temperature",
      type=float,
      default=1.15,
      help="temperature",
  )
  parser.add_argument(
      "-num_samples",
      type=int,
      default=30,
      help="number of samples to generate",
  )
  parser.add_argument(
      "-num_target_measure",
      type=int,
      default=4,
      help="number of target measures for conditioned generation",
  )
  parser.add_argument(
      "-choose_selected_tunes",
      action='store_true',
      help="generate samples from selected tunes, only for SOD dataset",
  )
  return parser

def main():
  args = get_argument_parser().parse_args()

  wandb_dir = Path('wandb')

  # check the exp_code exists
  exp_code = args.wandb_exp_dir
  if not (wandb_dir / exp_code).exists():
    print(f"Experiment with code {exp_code} does not exist.")
    return None
  
  ckpt_path, config_path, metadata_path, vocab_path = get_best_ckpt_path_and_config(wandb_dir, exp_code)
  config = OmegaConf.load(config_path)
  config = wandb_style_config_to_omega_config(config)
  print(ckpt_path)

  ckpt = torch.load(ckpt_path, map_location='cpu')
  model, test_set, vocab = prepare_model_and_dataset_from_config(config, metadata_path=metadata_path, vocab_path=vocab_path)

  # Load the model
  model.load_state_dict(ckpt['model'])
  model = model.eval()

  # prepare the dataset
  condition_list = [x[1] for x in test_set.data_list] 
  dataset_for_prompt = []
  for i in range(len(condition_list)):
    condition = test_set.get_segments_with_tune_idx(condition_list[i], 0)[0]
    dataset_for_prompt.append((condition, condition_list[i]))

  evaluator = Evaluator(config, model, dataset_for_prompt, vocab, device='cuda')

  # generate samples
  # apply sampling method
  sampling_method = args.sampling_method
  threshold = args.threshold
  temperature = args.temperature

  # conditioned generation
  if args.generation_type == 'conditioned':
    generated_sample_path = wandb_dir / exp_code / f'generated_samples_with_{args.num_target_measure}_measures_prompt_{sampling_method}_threshold_{threshold}_temperature_{temperature}'
    generated_sample_path.mkdir(parents=True, exist_ok=True)

    # prepare prompt and ground truth
    ground_truth_path = wandb_dir / exp_code / f'ground_truth_with_{args.num_target_measure}_measures'
    ground_truth_path.mkdir(parents=True, exist_ok=True)
    # evaluator.prepare_prompt_and_ground_truth(ground_truth_path, args.num_samples, args.num_target_measure)

    # generate samples
    # our selected prompt list
    if args.choose_selected_tunes and config.dataset == 'SOD':
      prompt_name_list = ['Requiem_orch', 'magnificat_bwv-243_8_orch', "Clarinet Concert in A Major: 2nd Movement, Adagio_orch", "symphony_098_3_orch", "symphony_38_504_3_orch", "Symphony in E Major, 3rd mvt Allegretto_orch", "string_quartet_17_2_orch",]
    else:
      prompt_name_list = [tune_name for _, tune_name in dataset_for_prompt]
      prompt_name_list = prompt_name_list[:args.num_samples]

    for tune_in_idx, tune_name in dataset_for_prompt:
      if tune_name in prompt_name_list:
        evaluator.generate_samples_with_prompt(generated_sample_path, args.num_target_measure, tune_in_idx, tune_name, sampling_method, threshold, temperature)
  # unconditioned generation
  else:
    generated_sample_path = wandb_dir / exp_code / f'generated_samples_unconditioned_{sampling_method}_threshold_{threshold}_temperature_{temperature}'
    generated_sample_path.mkdir(parents=True, exist_ok=True)
    evaluator.generate_samples_unconditioned(generated_sample_path, args.num_samples, sampling_method, threshold, temperature)

if __name__ == "__main__":
  main()
