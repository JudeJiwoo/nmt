from collections import defaultdict
from typing import Union
from math import log

import torch
from tqdm.auto import tqdm

from omegaconf import DictConfig
from .model_zoo import DoubleSequentialTransformer
from .symbolic_encoding.data_utils import TuneCompiler
from .symbolic_encoding.compile_utils import shift_and_pad
from .symbolic_encoding.compile_utils import reverse_shift_and_pad_for_tensor
from .encodec.vocab_utils import EncodecVocab
from .utils import add_conti_in_valid
from data_representation.vocab_utils import LangTokenVocab

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
               model:DoubleSequentialTransformer, 
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

    self.is_multiclass = True if config.data_params.encoding_scheme == 'nb' or config.data_params.encoding_scheme == 'cp' else False
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
