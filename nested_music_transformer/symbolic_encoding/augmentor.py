import random
from typing import Union

import torch

class Augmentor:
  def __init__(
      self, 
      vocab, 
      aug_type:Union[str, None], 
      input_length:int
  ):
    self.vocab = vocab
    self.aug_type = aug_type
    self.input_length = input_length
    self.feature_list = vocab.feature_list
    self.num_features = len(self.feature_list)
    self.encoding_scheme = vocab.encoding_scheme

    self.pitch_idx = self.feature_list.index('pitch')
    if 'chord' in self.feature_list:
      self.chord_idx = self.feature_list.index('chord')
  
  def _get_shift(self, segment):
    # the pitch vocab has ignore token in 0 index
    if self.encoding_scheme == 'cp' or self.encoding_scheme == 'nb':
      pitch_mask = segment != 0
      pitch_segment = segment[pitch_mask[:,self.pitch_idx], self.pitch_idx]
      # check if tensor is empty
      if pitch_segment.numel() == 0:
        shift = 0
      else:
        lowest_pitch = max(12, torch.min(pitch_segment))
        highest_pitch = min(119, torch.max(pitch_segment))
        lower_shift_bound = torch.where(lowest_pitch - torch.arange(6) > 11)[0][-1].item()
        upper_shift_bound = torch.where(highest_pitch + torch.arange(7) < 120)[0][-1].item()
        shift = random.randint(-lower_shift_bound, upper_shift_bound)
    else: # remi
      mask_for_pitch = self.vocab.total_mask['pitch'].to(segment.device)
      segemnt_pitch_mask = mask_for_pitch[segment]
      segment_pitch = segment * segemnt_pitch_mask
      segment_pitch = segment_pitch[segment_pitch != 0]
      # check if tensor is empty
      if segment_pitch.numel() == 0:
        shift = 0
      else:
        lower_bound = torch.argwhere(mask_for_pitch == 1)[0].item()
        upper_bound = torch.argwhere(mask_for_pitch == 1)[-1].item()
        lowest_pitch = max(lower_bound, torch.min(segment_pitch))
        highest_pitch = min(upper_bound, torch.max(segment_pitch))
        lower_shift_bound = torch.where(lowest_pitch - torch.arange(6) >= lower_bound)[0][-1].item()
        upper_shift_bound = torch.where(highest_pitch + torch.arange(7) <= upper_bound)[0][-1].item()
        shift = random.randint(-lower_shift_bound, upper_shift_bound)
    return shift

  # TODO: arrange hard coded part
  def __call__(self, segment):
    '''
    input_tensor is segments of x, y
    for transformer_xl, the shape of x, y is [max_num_segments, input_length, num_features]
    so we need to change the shape of x, y to [max_num_segments*input_length, num_features]
    '''
    if self.aug_type == 'random':
      shift = self._get_shift(segment)
      if self.encoding_scheme == 'cp' or self.encoding_scheme == 'nb':
        # pitch augmentation
        segment_pitch_mask = segment != 0
        new_segment = segment.clone()
        new_segment[segment_pitch_mask[:,self.pitch_idx], self.pitch_idx] += shift
        if 'chord' in self.feature_list:
          # chord augmentation
          segment_chord_mask = (segment[:,self.chord_idx] != 0) & (segment[:,self.chord_idx] != 1)
          new_segment[segment_chord_mask, self.chord_idx] = (((new_segment[segment_chord_mask, self.chord_idx]-2) %  12) + shift ) % 12 + ((new_segment[segment_chord_mask, self.chord_idx]-2) // 12) * 12 + 2
        segment = new_segment
      else: # remi
        # choose random interger between -5 and 6
        # the augmented results from shift -6 and 6 are same, so we choose -5 and 6
        # pitch augmentation
        mask_for_pitch = self.vocab.total_mask['pitch'].to(segment.device)
        segment_pitch_mask = mask_for_pitch[segment]
        new_segment = segment.clone()
        new_segment_valid = (new_segment + shift) * segment_pitch_mask
        new_segment = new_segment * (1 - segment_pitch_mask) + new_segment_valid
        if 'chord' in self.feature_list:
          # chord augmentation
          mask_for_chord = self.vocab.total_mask['chord'].clone().to(segment.device)
          chord_n_n_idx = torch.argwhere(mask_for_chord == 1)[-1].item()
          mask_for_chord[chord_n_n_idx] = 0
          start_idx_chord = self.vocab.remi_vocab_boundaries_by_key['chord'][0]
          segment_chord_mask = mask_for_chord[segment]
          new_segment_valid = ((((new_segment - start_idx_chord) % 12 + shift) % 12) + ((new_segment - start_idx_chord) // 12) * 12 + start_idx_chord) * segment_chord_mask
          new_segment = new_segment * (1 - segment_chord_mask) + new_segment_valid
        segment = new_segment
    return segment
