import os
import json
from pathlib import Path
from typing import Union

import torch

class EncodecVocab():
  def __init__(
      self, 
      in_vocab_file_path:Union[Path, None],
      encodec_tokens: list, 
      encoding_scheme:str,
  ):
    self.encodec_tokens_list = encodec_tokens
    self.encoding_scheme = encoding_scheme
    self._prepare_in_vocab(in_vocab_file_path)
    self._get_features()
    self._prepare_encodec_vocab()
    self._get_sos_eos_token()

  def _prepare_in_vocab(self, in_vocab_file_path):
    if os.path.isfile(in_vocab_file_path):
      self.in_vocab = json.load(open(in_vocab_file_path))
    else:
      self.in_vocab = None

  def _get_features(self):
    self.feature_list = ["k1", "k2", "k3", "k4"]

  def _prepare_encodec_vocab(self):
    if self.in_vocab is None:
      self.idx2event = {}
      self.event2idx = {}
      codebook_list = ['k1', 'k2', 'k3', 'k4']
      for codebook in codebook_list:
        token_list = list(range(0, 2049)) # 0~2048
        self.idx2event[codebook] = {int(idx): event for idx, event in enumerate(token_list)}
        self.event2idx[codebook] = {event: int(idx) for idx, event in enumerate(token_list)}
    else:
      self.idx2event = self.in_vocab
      self.event2idx = {key: {event: int(idx) for idx, event in enumerate(self.idx2event[key])} for key in self.feature_list}
  
  def _get_sos_eos_token(self):
    if self.encoding_scheme == 'remi':
      self.sos_token = [2048]
      self.eos_token = [[2048]]
    else:
      self.sos_token = [[2048, 2048, 2048, 2048]]
      self.eos_token = [[2048, 2048, 2048, 2048]]

  def save_vocab(self, json_path):
    with open(json_path, 'w') as f:
      json.dump(self.idx2event, f, indent=2, ensure_ascii=False)
  
  def get_vocab_size(self):
    return {key: len(self.idx2event[key]) for key in self.feature_list}
  
  def __call__(self, events):
    total_events = []
    for _, event in enumerate(events): 
      codebook_elements = []
      for element in event:
        codebook_elements.append(element.item())
      total_events.append(codebook_elements)
    total_events = torch.LongTensor(total_events)
    return total_events # [4, 1500]




