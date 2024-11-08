from collections import defaultdict
from typing import Union
from math import log
from omegaconf import DictConfig
from pathlib import Path
import pickle

import torch
from tqdm.auto import tqdm

from . import model_zoo
from .encodec.data_utils import EncodecDataset
from .symbolic_encoding import data_utils
from .model_zoo import NestedMusicTransformer
from .symbolic_encoding.data_utils import TuneCompiler
from .symbolic_encoding.compile_utils import shift_and_pad
from .symbolic_encoding.compile_utils import reverse_shift_and_pad_for_tensor
from .symbolic_encoding import decoding_utils
from .encodec.vocab_utils import EncodecVocab
from .train_utils import adjust_prediction_order
from data_representation import vocab_utils
from data_representation.vocab_utils import LangTokenVocab

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

  config_path = dir / 'files'  / 'config.yaml'
  vocab_path = next(ckpt_dir.glob('vocab*'))
  metadata_path = next(ckpt_dir.glob('*metadata.json'))

  # if there is pt file ending with 'last', return it 
  if len(list(ckpt_dir.glob('*last.pt'))) > 0:
    last_ckpt_fn = next(ckpt_dir.glob('*last.pt'))
  else:
    pt_fns = sorted(list(ckpt_dir.glob('*.pt')), key=lambda fn: int(fn.stem.split('_')[0].replace('iter', '')))
    last_ckpt_fn = pt_fns[-1]

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
    encoding_scheme = config.nn_params.encoding_scheme
    num_features = config.nn_params.num_features
    
    # get vocab
    vocab_name = {'remi':'LangTokenVocab', 'cp':'MusicTokenVocabCP', 'nb':'MusicTokenVocabNB'}
    selected_vocab_name = vocab_name[encoding_scheme]

    vocab = getattr(vocab_utils, selected_vocab_name)(
      in_vocab_file_path=vocab_path,
      event_data=None,
      encoding_scheme=encoding_scheme, 
      num_features=num_features)

    # Initialize symbolic dataset based on dataset name and configuration parameters
    symbolic_dataset = getattr(data_utils, dataset_name)(
                                vocab=vocab,
                                encoding_scheme=encoding_scheme,
                                num_features=num_features,
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

    # get proper prediction order according to the encoding scheme and target feature in the config
    prediction_order = adjust_prediction_order(encoding_scheme, num_features, config.data_params.first_pred_feature, nn_params)

    # Create the Transformer model based on configuration parameters
    nested_music_transformer = getattr(model_zoo, nn_params.model_name)(
                          vocab=symbolic_dataset.vocab,
                          input_length=config.train_params.input_length,
                          prediction_order=prediction_order,
                          input_embedder_name=nn_params.input_embedder_name,
                          main_decoder_name=nn_params.main_decoder_name,
                          sub_decoder_name=nn_params.sub_decoder_name,
                          sub_decoder_depth=nn_params.sub_decoder.num_layer if hasattr(nn_params, 'sub_decoder') else 0,
                          sub_decoder_enricher_use=nn_params.sub_decoder.feature_enricher_use \
                            if hasattr(nn_params, 'sub_decoder') and hasattr(nn_params.sub_decoder, 'feature_enricher_use') else False,
                          dim=nn_params.main_decoder.dim_model,
                          heads=nn_params.main_decoder.num_head,
                          depth=nn_params.main_decoder.num_layer,
                          dropout=nn_params.model_dropout,
                          )
    
  return nested_music_transformer, test_set, symbolic_dataset.vocab

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

# TODO: hard coded
def add_conti(list_of_lists, encoding_scheme):
  if encoding_scheme == 'nb':
    if len(list_of_lists[0]) == 4:
      # type, beat, pitch, duration
      for i in range(0, len(list_of_lists)):
        if list_of_lists[i][0] == 'SSS':
          list_of_lists[i][1] = 'Conti'
    elif len(list_of_lists[0]) == 5:
      # type, beat, instrument, pitch, duration
      previous_instrument = None
      for i in range(0, len(list_of_lists)):
        if list_of_lists[i][0] == 'SSS':
          list_of_lists[i][1] = 'Conti'
        if list_of_lists[i][2] == previous_instrument and previous_instrument != 0:
          list_of_lists[i][2] = 'Conti'
        else:
          previous_instrument = list_of_lists[i][2]
    elif len(list_of_lists[0]) == 7:
      # type, beat, chord, tempo, pitch, duration, velocity
      previous_chord = None
      previous_tempo = None
      for i in range(0, len(list_of_lists)):
        if list_of_lists[i][0] == 'SSS':
          list_of_lists[i][1] = 'Conti'
        if list_of_lists[i][2] == previous_chord and previous_chord != 0:
          list_of_lists[i][2] = 'Conti'
        elif list_of_lists[i][2] != previous_chord and list_of_lists[i][2] != 0:
          previous_chord = list_of_lists[i][2]
        if list_of_lists[i][3] == previous_tempo and previous_tempo != 0:
          list_of_lists[i][3] = 'Conti'
        elif list_of_lists[i][3] != previous_tempo and list_of_lists[i][3] != 0:
          previous_tempo = list_of_lists[i][3]
  elif encoding_scheme == 'cp':
    if len(list_of_lists[0]) == 7:
      # type, beat, chord, tempo, pitch, duration, velocity
      previous_chord = None
      previous_tempo = None
      for i in range(0, len(list_of_lists)):
        current_chord = list_of_lists[i][2]
        current_tempo = list_of_lists[i][3]
        if current_chord == previous_chord and current_chord != 0:
          list_of_lists[i][2] = 'Conti'
        elif current_chord != previous_chord and current_chord != 0:
          previous_chord = current_chord
        if current_tempo == previous_tempo and current_tempo != 0:
          list_of_lists[i][3] = 'Conti'
        elif current_tempo != previous_tempo and current_tempo != 0:
          previous_tempo = current_tempo
    if len(list_of_lists[0]) == 5:
      # type, beat, instrument, pitch, duration
      previous_instrument = None
      for i in range(0, len(list_of_lists)):
        current_instrument = list_of_lists[i][2]
        if current_instrument == previous_instrument and current_instrument != 0:
          list_of_lists[i][2] = 'Conti'
        elif current_instrument != previous_instrument and current_instrument != 0:
          previous_instrument = current_instrument
  return list_of_lists

class Evaluator:
  def __init__(self, 
               config: DictConfig, 
               model:NestedMusicTransformer, 
               test_set:TuneCompiler, 
               vocab: Union[LangTokenVocab, EncodecVocab],
               device:str='cuda',
               batch_size:int=16):
    self.config = config
    self.device = device
    self.vocab = vocab
    
    self.model = model
    self.model.eval()
    self.model.to(device)
    self.test_set = test_set
    
    self.input_len = config.train_params.input_length
    self.loss_by_class = {key:[] for key in self.vocab.feature_list}
    self.count_by_class = {key:0 for key in self.vocab.feature_list}
    self.batch_size = batch_size

    self.is_multiclass = True if config.nn_params.encoding_scheme == 'nb' or config.nn_params.encoding_scheme == 'cp' else False
    self.first_pred_feature = self.config.data_params.first_pred_feature

    self.neglect_keywords = ['SSS', 'SSN', 'Conti', 'Metrical', 'Note']
    self.valid_item_prob = []

    # we don't use focal loss on evaluation
    self.focal_alpha = 1
    self.focal_gamma = 0

  def save_results(self, save_fn):
    # convert loss_by_clas tensor to cpu
    for key in self.loss_by_class.keys():
      self.loss_by_class[key] = torch.tensor(self.loss_by_class[key]).cpu()
      self.count_by_class[key] = torch.tensor(self.count_by_class[key]).cpu()
    torch.save({'loss_by_class':self.loss_by_class, 'count_by_class':self.count_by_class}, save_fn)

  @torch.inference_mode()
  def get_perplexity(self):
    for data in tqdm(self.test_set.data_list, desc='Cal over dataset', position=0):
      data_tensor = torch.LongTensor(data[0])
      if self.config.data_params.encoding_scheme == 'nb':
        data_tensor = shift_and_pad(data_tensor, self.first_pred_feature)
        data_tensor = data_tensor[:-1]

      x_seg = data_tensor[:-1].unsqueeze(0)
      y_seg = data_tensor[1:].unsqueeze(0)
      self._cal_initial_seg(x_seg, y_seg)

      if x_seg.shape[1] > self.input_len:
        cat_logits = []
        cat_y = []
        batch_x = x_seg[0, 1:].unfold(dimension=0, size=self.input_len, step=1)
        batch_y = y_seg[0, 1:].unfold(dimension=0, size=self.input_len, step=1)
        if self.is_multiclass:
          batch_x = batch_x.transpose(1,2)
          batch_y = batch_y.transpose(1,2)
        for batch_start_idx in tqdm(range(0, batch_x.shape[0], self.batch_size), desc='In piece iter', position=1, leave=False):
          x = batch_x[batch_start_idx:batch_start_idx+self.batch_size]
          y = batch_y[batch_start_idx:batch_start_idx+self.batch_size]
          logits, y = self._cal_following_seg(x, y)
          cat_logits.append(logits)
          cat_y.append(y)
        if self.is_multiclass:
          cat_dict = {}
          for key in self.vocab.feature_list:
            cat_dict[key] = torch.cat([logits_dict[key] for logits_dict in cat_logits], dim=0)
          cat_logits = cat_dict
        else:
          cat_logits = torch.cat(cat_logits, dim=0)
        cat_y = torch.cat(cat_y, dim=0)
        if self.is_multiclass:
          self._update_loss_for_multi_class(cat_logits, cat_y)
        else:
          cat_prob = torch.nn.functional.softmax(cat_logits, dim=-1)
          pt = cat_prob[torch.arange(cat_prob.shape[0]), cat_y]
          # focal_loss = -self.focal_alpha * (1-pt)**self.focal_gamma * torch.log(pt) # [batch_size*seq_len]
          loss = -torch.log(pt)
          self._update_loss_for_single_class(loss, cat_y)

  @torch.inference_mode()
  def _update_loss_for_single_class(self, neg_log_prob:torch.Tensor, y:torch.Tensor):
    for key in self.vocab.feature_list:
      feature_mask = self.vocab.total_mask[key].to(y.device) # [vocab_size,]
      mask_for_target = feature_mask[y] # [b*t]
      normal_loss_seq_by_class = neg_log_prob[mask_for_target==1]
      if mask_for_target.sum().item() != 0:
        self.loss_by_class[key] += normal_loss_seq_by_class.tolist()
        self.count_by_class[key] += mask_for_target.sum().item()

  @torch.inference_mode()
  def _update_loss_for_multi_class(self, logits_dict:dict, tgt:torch.Tensor):
    correct_token_prob = []
    for index, key in enumerate(self.vocab.feature_list):
      logit_values = logits_dict[key]
      prob_values = torch.nn.functional.softmax(logit_values, dim=-1)
      correct_token_prob.append(prob_values[torch.arange(prob_values.shape[0]), tgt[:, index]])
    correct_token_prob = torch.stack(correct_token_prob, dim=1)
    # tgt = reverse_shift_and_pad_for_tensor(tgt, self.first_pred_feature)
    y_decoded = self.vocab.decode(tgt)
    y_decoded = add_conti(y_decoded, self.config.data_params.encoding_scheme)
    # correct_token_prob = reverse_shift_and_pad_for_tensor(correct_token_prob, self.first_pred_feature)
    num_notes = logits_dict['pitch'].shape[0]
    cum_prob = 1
    for idx in range(num_notes):
      token = y_decoded[idx]
      token_prob = correct_token_prob[idx].tolist()
      for j, key in enumerate(self.vocab.feature_list):
        cur_feature = token[j]
        # clamp cur_prob to avoid when cur_prob is 0
        cur_prob = max(token_prob[j], 1e-10)
        if cur_feature == 0: # ignore token
          continue
        if cur_feature in self.neglect_keywords:
          cum_prob *= cur_prob
          continue
        if self.config.data_params.encoding_scheme == 'cp' and 'time_signature' in cur_feature:
          cum_prob *= cur_prob
          continue
        if self.config.data_params.encoding_scheme == 'cp' and 'Bar' in cur_feature:
          cum_prob = 1
          continue
        self.valid_item_prob.append([cur_feature, cur_prob, cur_prob*cum_prob])
        pt = cur_prob*cum_prob
        loss = -log(pt)
        self.loss_by_class[key].append(loss)
        self.count_by_class[key] += 1
        cum_prob = 1

  @torch.inference_mode()
  def _cal_initial_seg(self, x_seg, y_seg):
    x, y = x_seg[:, :self.input_len].to(self.device), y_seg[:, :self.input_len].to(self.device)
    logits = self.model(x, y)
    y = y.flatten(0,1)
    if self.is_multiclass:
      for key in logits.keys():
        logits[key] = logits[key].flatten(0,1)
      self._update_loss_for_multi_class(logits, y)
    else:
      prob = torch.nn.functional.softmax(logits, dim=-1)
      prob = prob.flatten(0,1)
      pt = prob[torch.arange(len(y)), y]
      loss = -torch.log(pt)
      self._update_loss_for_single_class(loss, y)

  @torch.inference_mode()
  def _cal_following_seg(self, x:torch.Tensor, y:torch.Tensor):
    x, y = x.to(self.device), y.to(self.device)
    logits = self.model(x, y)
    y = y[:, -1:].flatten(0,1).cpu()
    if self.is_multiclass:
      logits_dict = {}
      for key in self.vocab.feature_list:
        logits_dict[key] = logits[key][:, -1:].flatten(0,1).cpu()
      return logits_dict, y
    else:
      logits = logits[:, -1:].flatten(0,1).cpu()
      return logits, y

  def prepare_prompt_and_ground_truth(self, save_dir, num_target_samples, num_target_measures):
    encoding_scheme = self.config.nn_params.encoding_scheme

    in_beat_resolution_dict = {'Pop1k7': 4, 'Pop909': 4, 'SOD': 12, 'LakhClean': 4}
    in_beat_resolution = in_beat_resolution_dict[self.config.dataset]

    midi_decoder_dict = {'remi':'MidiDecoder4REMI', 'cp':'MidiDecoder4CP', 'nb':'MidiDecoder4NB'}
    decoder_name = midi_decoder_dict[encoding_scheme]
    decoder = getattr(decoding_utils, decoder_name)(vocab=self.vocab, in_beat_resolution=in_beat_resolution, dataset_name=self.config.dataset)

    for i, (tuneidx, tune_name) in enumerate(self.test_set):
      ground_truth_sample = tuneidx
      try:
        decoder(ground_truth_sample, output_path=str(save_dir / f"{i}_{tune_name}_gt.mid"))
      except:
        print(f"Error in generating {i}_{tune_name}.mid")

      prompt = self.model.decoder._prepare_inference(start_token=self.model.decoder.net.start_token, manual_seed=0, condition=tuneidx, num_target_measures=num_target_measures)
      try:
        decoder(prompt, output_path=str(save_dir / f"{i}_{tune_name}_prompt.mid"))
      except:
        print(f"Error in generating {i}_{tune_name}_prompt.mid")

      if i == num_target_samples:
        break

  def generate_samples_with_prompt(self, save_dir, num_target_measures, tuneidx, tune_name, sampling_method=None, threshold=None, temperature=1.0):
    encoding_scheme = self.config.nn_params.encoding_scheme

    in_beat_resolution_dict = {'Pop1k7': 4, 'Pop909': 4, 'SOD': 12, 'LakhClean': 4}
    in_beat_resolution = in_beat_resolution_dict[self.config.dataset]

    midi_decoder_dict = {'remi':'MidiDecoder4REMI', 'cp':'MidiDecoder4CP', 'nb':'MidiDecoder4NB'}
    decoder_name = midi_decoder_dict[encoding_scheme]
    decoder = getattr(decoding_utils, decoder_name)(vocab=self.vocab, in_beat_resolution=in_beat_resolution, dataset_name=self.config.dataset)

    tuneidx = tuneidx.cuda()
    generated_sample = self.model.generate(0, self.input_len, condition=tuneidx, num_target_measures=num_target_measures, sampling_method=sampling_method, threshold=threshold, temperature=temperature)
    decoder(generated_sample, output_path=str(save_dir / f"{tune_name}.mid"))

    prompt = self.model.decoder._prepare_inference(self.model.decoder.net.start_token, 0, tuneidx, num_target_measures=8)
    decoder(prompt, output_path=str(save_dir / f"{tune_name}_prompt.mid"))

  def generate_samples_unconditioned(self, save_dir, num_samples, sampling_method, threshold, temperature):
    encoding_scheme = self.config.nn_params.encoding_scheme

    in_beat_resolution_dict = {'Pop1k7': 4, 'Pop909': 4, 'SOD': 12, 'LakhClean': 4}
    in_beat_resolution = in_beat_resolution_dict[self.config.dataset]

    midi_decoder_dict = {'remi':'MidiDecoder4REMI', 'cp':'MidiDecoder4CP', 'nb':'MidiDecoder4NB'}
    decoder_name = midi_decoder_dict[encoding_scheme]
    decoder = getattr(decoding_utils, decoder_name)(vocab=self.vocab, in_beat_resolution=in_beat_resolution, dataset_name=self.config.dataset)

    for i in range(num_samples):
      generated_sample = self.model.generate(0, self.input_len, condition=None, num_target_measures=None, sampling_method=sampling_method, threshold=threshold, temperature=temperature)
      decoder(generated_sample, output_path=str(save_dir / f"{i}.mid"))
