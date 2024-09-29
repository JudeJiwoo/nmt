import json
from pathlib import Path
from typing import Union

import torch

from .vocab_utils import EncodecVocab

def delay_tensor(tensor, pad_value=2048):
  tensor_shape = tensor.shape
  result = torch.ones(tensor_shape, dtype=torch.long) * pad_value
  for i in range(tensor_shape[1]):
    result[i:, i] = tensor[:tensor_shape[0]-(i), i]
  return result

class TuneCompiler():
  def __init__(
      self, 
      encodec_tokens, 
      input_length, 
      encoding_scheme
  ):
    self.segments = self._make_segments(encodec_tokens, input_length, encoding_scheme)

  def _make_segments(self, data_list, input_length, encoding_scheme):
    segments = []
    for _, tokens in data_list:
      for key, value in tokens.items():
        if type(key) == str:
          continue
        tune_in_idx = value.squeeze(0).T # (seq_len, 4)
        bos_token = torch.LongTensor([2048, 2048, 2048, 2048]).unsqueeze(0)
        tune_in_idx = torch.cat([bos_token, tune_in_idx], dim=0)
        if encoding_scheme == "remi":
          x = tune_in_idx.flatten(0,1)[3:-1]
          y = tune_in_idx.flatten(0,1)[4:]
        elif encoding_scheme == "nb":
          x = tune_in_idx[:-1]
          y = tune_in_idx[1:]
        elif encoding_scheme == "nb_delay":
          delayed_tune_in_idx = delay_tensor(tune_in_idx)
          x = delayed_tune_in_idx[:-1]
          y = delayed_tune_in_idx[1:]
        for i in range(0, len(y), input_length):
          segment_x = x[i:i+input_length]
          segment_y = y[i:i+input_length]
          if len(segment_x) < input_length or len(segment_y) < input_length:
            # EOS will be 2048
            if encoding_scheme == "nb" or encoding_scheme == "nb_delay":
              pad_x = torch.zeros((input_length-len(segment_x), 4), dtype=torch.long) + 2048
              pad_y = torch.zeros((input_length-len(segment_y), 4), dtype=torch.long) + 2048
            elif encoding_scheme == "remi":
              pad_x = torch.zeros((input_length-len(segment_x)), dtype=torch.long) + 2048
              pad_y = torch.zeros((input_length-len(segment_y)), dtype=torch.long) + 2048
            segment_x = torch.cat([segment_x, pad_x], dim=0)
            segment_y = torch.cat([segment_y, pad_y], dim=0)
            mask = torch.cat([torch.ones(len(segment_x)-len(pad_x), dtype=torch.long), torch.zeros(len(pad_x), dtype=torch.long)], dim=0)
          else:
            mask = torch.ones(input_length, dtype=torch.long)
          segments.append([segment_x.long(), segment_y.long(), mask.long()])
    return segments

  def __len__(self):
    return len(self.segments)

  def __getitem__(self, idx):
    return self.segments[idx]

class EncodecDataset():
  def __init__(
      self, 
      in_vocab_file_path:Union[Path, None],
      out_vocab_path:Union[Path, None],
      encoding_scheme:str, 
      input_length:int,
      token_path:Path
  ):
    self.input_length = input_length
    self.encoding_scheme = encoding_scheme

    self._prepare_encodec_tokens(token_path)

    self.vocab = EncodecVocab(in_vocab_file_path, self.encodec_tokens_list, encoding_scheme)
    if out_vocab_path is not None:
      self.vocab.save_vocab(out_vocab_path)

  def _prepare_encodec_tokens(self, token_path):
    encodec_tokens_path = Path(token_path)
    self.encodec_tokens_list = sorted(list(encodec_tokens_path.rglob('*.pt')))
    self.encodec_tokens_list = [(encodec_token.stem, torch.load(encodec_token)) for encodec_token in self.encodec_tokens_list]

  def _extract_indexes(self, meta_data):
    train_indexes = []
    validation_indexes = []
    test_indexes = []
    for index, tag in meta_data.items():
      if tag == 'train':
        train_indexes.append(int(index))
      elif tag == 'validation':
        validation_indexes.append(int(index))
      elif tag == 'test':
        test_indexes.append(int(index))
    return train_indexes, validation_indexes, test_indexes

  def split_train_valid_test_set(self):
    train_data = []
    valid_data = []
    self.test_data = []
    with open("metadata/maestro-v3.0.0.json", "r") as f:
      maestro_metadata = json.load(f)
    train_indexes, validation_indexes, test_indexes = self._extract_indexes(meta_data=maestro_metadata['split'])
    file_name_idx_dict = {file_name.split('/')[-1]:int(idx) for idx, file_name in maestro_metadata['midi_filename'].items()}
    if "semcodec" in self.encodec_tokens_list[0][0]:
      semcodec = True
      finetune = False
    elif "finetuned" in self.encodec_tokens_list[0][0]:
      semcodec = False
      finetune = True
    else:
      semcodec = False
      finetune = False
    for file_name, tokens in self.encodec_tokens_list:
      if semcodec:
        file_name = file_name.replace("_32khz_mono_semcodec", "")
      elif finetune:
        file_name = file_name.replace("_32khz_mono_encodec_finetuned", "")
      tune_idx = file_name_idx_dict[file_name+'.midi']
      if tune_idx in train_indexes:
        train_data.append((file_name, tokens))
      elif tune_idx in validation_indexes:
        valid_data.append((file_name, tokens))
      elif tune_idx in test_indexes:
        self.test_data.append((file_name, tokens))
      else:
        raise ValueError(f"tune_idx {tune_idx} not in train, valid, or test")
    train_dataset = TuneCompiler(train_data, self.input_length, self.encoding_scheme)
    valid_dataset = TuneCompiler(valid_data, self.input_length, self.encoding_scheme)
    test_dataset = TuneCompiler(self.test_data, self.input_length, self.encoding_scheme)
    return train_dataset, valid_dataset, test_dataset

if __name__ == "__main__":
  in_vocab_path = "vocab/MaestroEncodec_vocab/maestro-v3.0.0-in_vocab.json"
  encodec_dataset = EncodecDataset(in_vocab_path, None, "nb", 1500)
  encodec_dataset[0]