import random
from collections import defaultdict

import torch
import numpy as np

def reverse_shift_and_pad(tune_in_idx, slice_boundary=4):
  new_lst = [curr_elems[:slice_boundary] + next_elems[slice_boundary:] for curr_elems, next_elems in zip(tune_in_idx, tune_in_idx[1:])]
  return new_lst

def reverse_shift_and_pad_for_tensor(tensor, first_pred_feature):
  '''
  tensor: [batch_size x seq_len x feature_size]
  '''
  if first_pred_feature == 'type':
    return tensor
  if tensor.shape[-1] == 8:
    slice_boundary_dict = {'type':0, 'beat':1, 'chord':2, 'tempo':3, 'instrument':4, 'pitch':5, 'duration':6, 'velocity':7}
  elif tensor.shape[-1] == 7:
    slice_boundary_dict = {'type':0, 'beat':1, 'chord':2, 'tempo':3, 'pitch':4, 'duration':5, 'velocity':6}
  elif tensor.shape[-1] == 5:
    slice_boundary_dict = {'type':0, 'beat':1, 'instrument':2, 'pitch':3, 'duration':4}
  elif tensor.shape[-1] == 4:
    slice_boundary_dict = {'type':0, 'beat':1, 'pitch':2, 'duration':3}
  slice_boundary = slice_boundary_dict[first_pred_feature]
  new_tensor = torch.zeros_like(tensor)
  new_tensor[..., :, :slice_boundary] = tensor[..., :, :slice_boundary]
  new_tensor[..., :-1, slice_boundary:] = tensor[..., 1:, slice_boundary:]
  return new_tensor

def shift_and_pad(tune_in_idx, first_pred_feature):
  if first_pred_feature == 'type':
    return tune_in_idx
  if len(tune_in_idx[0]) == 8:
    slice_boundary_dict = {'type':0, 'beat':-7, 'chord':-6, 'tempo':-5, 'instrument':-4, 'pitch':-3, 'duration':-2, 'velocity':-1}
  elif len(tune_in_idx[0]) == 7:
    slice_boundary_dict = {'type':0, 'beat':-6, 'chord':-5, 'tempo':-4, 'pitch':-3, 'duration':-2, 'velocity':-1}
  elif len(tune_in_idx[0]) == 5:
    slice_boundary_dict = {'type':0, 'beat':-4, 'instrument':-3, 'pitch':-2, 'duration':-1}
  elif len(tune_in_idx[0]) == 4: 
    slice_boundary_dict = {'type':0, 'beat':-3, 'pitch':-2, 'duration':-1}
  slice_boundary = slice_boundary_dict[first_pred_feature]
  # Add an empty list padded with zeros at the beginning, and sos and eos tokens are not shifted
  padded_tune_in_idx = torch.cat([torch.zeros(1, len(tune_in_idx[0]), dtype=torch.long), tune_in_idx], dim=0)
  new_tensor = torch.zeros_like(padded_tune_in_idx)
  new_tensor[:, slice_boundary:] = padded_tune_in_idx[:, slice_boundary:]
  new_tensor[:-1, :slice_boundary] = padded_tune_in_idx[1:, :slice_boundary]
  return new_tensor

class VanillaTransformer_compiler():
  def __init__(
      self, 
      data_list, 
      augmentor, 
      eos_token, 
      input_length,
      first_pred_feature,
      encoding_scheme
  ):
    self.data_list = data_list
    self.augmentor = augmentor
    self.eos_token = eos_token
    self.input_length = input_length
    self.first_pred_feature = first_pred_feature
    self.encoding_scheme = encoding_scheme

  def make_segments(self, data_type):
    segments = []
    tune_name2segment = defaultdict(list)
    segment2tune_name = []
    num_segments = 0
    for i in range(len(self.data_list)):
      tune_in_idx, tune_name = self.data_list[i]
      tune_in_idx = torch.LongTensor(tune_in_idx)
      if self.encoding_scheme == 'remi':
        eos_token = torch.LongTensor(self.eos_token)
      else:
        eos_token = torch.LongTensor(self.eos_token)
        # shift and pad
        tune_in_idx = shift_and_pad(tune_in_idx, self.first_pred_feature)
      if data_type == 'train':
        if len(tune_in_idx) <= self.input_length+1:
          if 'remi' in self.encoding_scheme:
            padding_seq = eos_token[0].repeat(self.input_length+1-len(tune_in_idx))
          else:
            padding_seq = eos_token.repeat(self.input_length+1-len(tune_in_idx), 1)
          mask = torch.cat([torch.ones(len(tune_in_idx), dtype=torch.long), torch.zeros(len(padding_seq), dtype=torch.long)], dim=0)
          segment = torch.cat([tune_in_idx, padding_seq], dim=0)
          segments.append([segment, mask])
          segment2tune_name.append(tune_name)
        else:
          start_point = 0
          while start_point + self.input_length+1 < len(tune_in_idx):
            mask = torch.ones(self.input_length+1, dtype=torch.long)
            segment = tune_in_idx[start_point:start_point + self.input_length+1]
            segments.append([segment, mask])
            segment2tune_name.append(tune_name)
            assert len(segment) == self.input_length+1
            # Randomly choose the start point for the next segment, which is in the range of half of the current segment to the end of the current segment
            start_point += random.randint((self.input_length+1)//2, self.input_length+1)
          # add the last segment
          if len(tune_in_idx[start_point:]) < self.input_length+1:
            if 'remi' in self.encoding_scheme:
              padding_seq = eos_token[0].repeat(self.input_length+1-len(tune_in_idx[start_point:]))
            else:
              padding_seq = eos_token.repeat(self.input_length+1-len(tune_in_idx[start_point:]), 1)
            mask = torch.cat([torch.ones(len(tune_in_idx[start_point:]), dtype=torch.long), torch.zeros(len(padding_seq), dtype=torch.long)], dim=0)
            segment = torch.cat([tune_in_idx[start_point:], padding_seq], dim=0)
            segments.append([segment, mask])
            segment2tune_name.append(tune_name)
      else: # for validset
        for i in range(0, len(tune_in_idx), self.input_length+1):
          segment = tune_in_idx[i:i+self.input_length+1]
          if len(segment) <= self.input_length+1:
            if 'remi' in self.encoding_scheme:
              padding_seq = eos_token[0].repeat(self.input_length+1-len(segment))
            else:
              padding_seq = eos_token.repeat(self.input_length+1-len(segment), 1)
            mask = torch.cat([torch.ones(len(segment), dtype=torch.long), torch.zeros(len(padding_seq), dtype=torch.long)], dim=0)
            segment = torch.cat([segment, padding_seq], dim=0)
            segment2tune_name.append(tune_name)
            segments.append([segment, mask])
            num_segments += 1
            tune_name2segment[tune_name].append(num_segments-1)
          else:
            mask = torch.ones(self.input_length+1, dtype=torch.long)
            segments.append([segment, mask])
            segment2tune_name.append(tune_name)
            segments.append([segment, mask])
            num_segments += 1
            tune_name2segment[tune_name].append(num_segments-1)
          assert len(segment) == self.input_length+1
    return segments, tune_name2segment, segment2tune_name
