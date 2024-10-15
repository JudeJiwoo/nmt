import re
import random
from pathlib import Path
from collections import OrderedDict
from typing import Union, List, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt

import json
from tqdm import tqdm

from .augmentor import Augmentor
from .compile_utils import VanillaTransformer_compiler
from data_representation import vocab_utils

def get_emb_total_size(config, vocab):
  emb_param = config.nn_params.emb
  total_size = 0 
  for feature in vocab.feature_list:
    size = int(emb_param[feature] * emb_param.emb_size)
    total_size += size
    emb_param[feature] = size
  emb_param.total_size = total_size
  config.nn_params.emb = emb_param
  return config

class TuneCompiler():
  def __init__(
      self, 
      data:List[Tuple[np.ndarray, str]], 
      data_type:str, 
      augmentor:Augmentor, 
      vocab,
      input_length:int,
      first_pred_feature:str
  ):
    '''
    if augmentor is None, it means that it is test dataset
    data_list has either list of tune_in_idx or list of (tune_in_idx, replaced_tune_in_idx)
    '''
    self.data_list = data
    self.data_type = data_type
    self.augmentor = augmentor
    self.eos_token = vocab.eos_token
    self.compile_function = VanillaTransformer_compiler(
      data_list=self.data_list, 
      augmentor=self.augmentor, 
      eos_token=self.eos_token, 
      input_length=input_length,
      first_pred_feature=first_pred_feature,
      encoding_scheme=vocab.encoding_scheme
    )
    if self.data_type == 'valid' or self.data_type == 'test':
      self._update_segments_for_validset()
    else:
      self._update_segments_for_trainset()

  def _update_segments_for_trainset(self, random_seed=0):
    random.seed(random_seed)
    self.segments, _, self.segment2tune_name = self.compile_function.make_segments(self.data_type)
    print(f"number of trainset segments: {len(self.segments)}")

  def _update_segments_for_validset(self):
    random.seed(0)
    self.segments, self.tune_name2segment, self.segment2tune_name = self.compile_function.make_segments(self.data_type)
    print(f"number of testset segments: {len(self.segments)}")

  def __getitem__(self, idx):
    segment, tensor_mask = self.segments[idx]
    tune_name = self.segment2tune_name[idx]
    if self.data_type == 'train':
      augmented_segment = self.augmentor(segment)
      return augmented_segment, tensor_mask, tune_name
    else:
      return segment, tensor_mask, tune_name
  
  def get_segments_with_tune_idx(self, tune_name, seg_order):
    '''
    for testset with conditional generation
    '''
    segments_list = self.tune_name2segment[tune_name]
    segment_idx = segments_list[seg_order]
    segment, mask = self.segments[segment_idx][0], self.segments[segment_idx][1]
    return segment, mask

  def __len__(self):
    return len(self.segments)

class SymbolicMusicDataset():
  def __init__(
      self, 
      vocab: vocab_utils.LangTokenVocab,
      encoding_scheme:str, 
      num_features:int, 
      debug:bool, 
      aug_type:Union[str, None], 
      input_length:int,
      first_pred_feature:str
  ):
    self.encoding_scheme = encoding_scheme
    self.num_features = num_features
    self.debug = debug
    self.input_length = input_length
    self.first_pred_feature = first_pred_feature

    # get vocab
    self.vocab = vocab
    
    # get augmentor
    self.augmentor = Augmentor(vocab=self.vocab, aug_type=aug_type, input_length=input_length)
    
    # load tune_in_idx
    self.tune_in_idx, self.len_tunes, self.file_name_list = self._load_tune_in_idx()
    
    # plot histogram of tune length
    dataset_name = self.__class__.__name__
    len_dir_path = Path(f"len_tunes/{dataset_name}")
    len_dir_path.mkdir(parents=True, exist_ok=True)
    self._plot_hist(self.len_tunes, len_dir_path / f"len_{encoding_scheme}{num_features}.png")

  def _load_tune_in_idx(self) -> Tuple[Dict[str, np.ndarray], Dict[str, int], List[str]]:
    print("preprocessed tune_in_idx data is being loaded")
    tune_in_idx_list = sorted(list(Path(f"dataset/represented_data/tuneidx/tuneidx_{self.__class__.__name__}/{self.encoding_scheme}{self.num_features}").rglob("*.npz")))
    if self.debug:
      tune_in_idx_list = tune_in_idx_list[:5000]
    tune_in_idx_dict = OrderedDict()
    len_tunes = OrderedDict()
    file_name_list = []
    for tune_in_idx_file in tqdm(tune_in_idx_list, total=len(tune_in_idx_list)):
      tune_in_idx = np.load(tune_in_idx_file)['arr_0']
      tune_in_idx_dict[tune_in_idx_file.stem] = tune_in_idx
      len_tunes[tune_in_idx_file.stem] = len(tune_in_idx)
      file_name_list.append(tune_in_idx_file.stem)
    return tune_in_idx_dict, len_tunes, file_name_list

  def _plot_hist(self, len_tunes, path_outfile):
    Path(path_outfile).parent.mkdir(parents=True, exist_ok=True)
    data = np.array(list(len_tunes.values()))
    self.mean_len_tunes = np.mean(data)
    data_mean = np.mean(data)
    data_std = np.std(data)
    plt.figure(dpi=100)
    plt.hist(data, bins=50)
    plt.title(f"mean: {data_mean:.2f}, std: {data_std:.2f}")
    plt.savefig(path_outfile)
    plt.close()

  def _get_split_list_from_tune_in_idx(self, ratio, seed):
    shuffled_tune_names = list(self.tune_in_idx.keys())
    random.seed(seed)
    random.shuffle(shuffled_tune_names)
    num_train = int(len(shuffled_tune_names)*ratio)
    num_valid = int(len(shuffled_tune_names)*(1-ratio)/2)

    train_names = shuffled_tune_names[:num_train]
    valid_names = shuffled_tune_names[num_train:num_train+num_valid]
    test_names = shuffled_tune_names[num_train+num_valid:]
    return train_names, valid_names, test_names, shuffled_tune_names

  def split_train_valid_test_set(self, dataset_name=None, ratio=None, seed=42, save_dir=None):
    if not Path(f"metadata/{dataset_name}_metadata.json").exists():
      assert ratio is not None, "ratio should be given when you make metadata for split"
      train_names, valid_names, test_names, shuffled_tune_names = self._get_split_list_from_tune_in_idx(ratio, seed)

      print(f"Randomly split train and test set using seed {seed}")
      out_dict = {'shuffle_seed':seed,
                  'shuffled_names': shuffled_tune_names, 
                  'train':train_names, 
                  'valid':valid_names, 
                  'test':test_names}
      with open(f"metadata/{dataset_name}_metadata.json", "w") as f:
        json.dump(out_dict, f, indent=2)
    else:
      with open(f"metadata/{dataset_name}_metadata.json", "r") as f:
        out_dict = json.load(f)
      train_names, valid_names, test_names = out_dict['train'], out_dict['valid'], out_dict['test']
      assert set(out_dict['shuffled_names']) == set(self.tune_in_idx.keys()), "Loaded data is not matched with the recorded metadata"

    train_data = [(self.tune_in_idx[tune_name], tune_name) for tune_name in train_names]
    valid_data = [(self.tune_in_idx[tune_name], tune_name) for tune_name in valid_names]
    self.test_data = [(self.tune_in_idx[tune_name], tune_name) for tune_name in test_names]

    # sort by length in the case of transformer_xl
    train_data.sort(key=lambda x: len(x[0]), reverse=True)
    valid_data.sort(key=lambda x: len(x[0]), reverse=True)
    # self.test_data.sort(key=lambda x: len(x[0]), reverse=True)

    train_dataset = TuneCompiler(data=train_data, data_type='train', augmentor=self.augmentor, vocab=self.vocab, input_length=self.input_length, first_pred_feature=self.first_pred_feature)
    valid_dataset = TuneCompiler(data=valid_data, data_type='valid', augmentor=self.augmentor, vocab=self.vocab, input_length=self.input_length, first_pred_feature=self.first_pred_feature)
    test_dataset = TuneCompiler(data=self.test_data, data_type='test', augmentor=self.augmentor, vocab=self.vocab, input_length=self.input_length, first_pred_feature=self.first_pred_feature)

    # save meata data
    if save_dir is not None:
      Path(save_dir).mkdir(parents=True, exist_ok=True)
      with open(Path(save_dir) / f"{dataset_name}_metadata.json", "w") as f:
        json.dump(out_dict, f, indent=2)
    return train_dataset, valid_dataset, test_dataset

class Pop1k7(SymbolicMusicDataset):
  def __init__(self, in_vocab_file_path, out_vocab_path, encoding_scheme, num_features, debug, aug_type, input_length, first_pred_feature):
    super().__init__(in_vocab_file_path, out_vocab_path, encoding_scheme, num_features, debug, aug_type, input_length, first_pred_feature)

class SymphonyMIDI(SymbolicMusicDataset):
  def __init__(self, in_vocab_file_path, out_vocab_path, encoding_scheme, num_features, debug, aug_type, input_length, first_pred_feature):
    super().__init__(in_vocab_file_path, out_vocab_path, encoding_scheme, num_features, debug, aug_type, input_length, first_pred_feature)

class LakhClean(SymbolicMusicDataset):
  def __init__(self, in_vocab_file_path, out_vocab_path, encoding_scheme, num_features, debug, aug_type, input_length, first_pred_feature):
    super().__init__(in_vocab_file_path, out_vocab_path, encoding_scheme, num_features, debug, aug_type, input_length, first_pred_feature)

  def _get_split_list_from_tune_in_idx(self, ratio, seed):
    shuffled_tune_names = list(self.tune_in_idx.keys())
    song_names_without_version = [re.sub(r"\.\d+$", "", song) for song in shuffled_tune_names]
    song_dict = {}
    for song, orig_song in zip(song_names_without_version, shuffled_tune_names):
      if song not in song_dict:
        song_dict[song] = []
      song_dict[song].append(orig_song)
    unique_song_names = list(song_dict.keys())
    random.seed(seed)
    random.shuffle(unique_song_names)
    num_train = int(len(unique_song_names)*ratio)
    num_valid = int(len(unique_song_names)*(1-ratio)/2)
    train_names = []
    valid_names = []
    test_names = []
    for song_name in unique_song_names[:num_train]:
      train_names.extend(song_dict[song_name])
    for song_name in unique_song_names[num_train:num_train+num_valid]:
      valid_names.extend(song_dict[song_name])
    for song_name in unique_song_names[num_train+num_valid:]:
      test_names.extend(song_dict[song_name])
    return train_names, valid_names, test_names, shuffled_tune_names

class SOD(SymbolicMusicDataset):
  def __init__(self, in_vocab_file_path, out_vocab_path, encoding_scheme, num_features, debug, aug_type, input_length, first_pred_feature):
    super().__init__(in_vocab_file_path, out_vocab_path, encoding_scheme, num_features, debug, aug_type, input_length, first_pred_feature)
  
  def _load_tune_in_idx(self) -> Tuple[Dict[str, np.ndarray], Dict[str, int], List[str]]:
    print("preprocessed tune_in_idx data is being loaded")
    tune_in_idx_list = sorted(list(Path(f"dataset/represented_data/tuneidx/tuneidx_{self.__class__.__name__}/{self.encoding_scheme}{self.num_features}").rglob("*.npz")))
    if self.debug:
      tune_in_idx_list = tune_in_idx_list[:5000]
    tune_in_idx_dict = OrderedDict()
    len_tunes = OrderedDict()
    file_name_list = []
    with open("metadata/SOD_irregular_tunes.json", "r") as f:
      irregular_tunes = json.load(f)
    for tune_in_idx_file in tqdm(tune_in_idx_list, total=len(tune_in_idx_list)):
      if tune_in_idx_file.stem in irregular_tunes:
        continue
      tune_in_idx = np.load(tune_in_idx_file)['arr_0']
      tune_in_idx_dict[tune_in_idx_file.stem] = tune_in_idx
      len_tunes[tune_in_idx_file.stem] = len(tune_in_idx)
      file_name_list.append(tune_in_idx_file.stem)
    print(f"number of loaded tunes: {len(tune_in_idx_dict)}")
    return tune_in_idx_dict, len_tunes, file_name_list

class BachChorale(SymbolicMusicDataset):
  def __init__(self, in_vocab_file_path, out_vocab_path, encoding_scheme, num_features, debug, aug_type, input_length, first_pred_feature):
    super().__init__(in_vocab_file_path, out_vocab_path, encoding_scheme, num_features, debug, aug_type, input_length, first_pred_feature)

class Pop909(SymbolicMusicDataset):
  def __init__(self, in_vocab_file_path, out_vocab_path, encoding_scheme, num_features, debug, aug_type, input_length, first_pred_feature):
    super().__init__(in_vocab_file_path, out_vocab_path, encoding_scheme, num_features, debug, aug_type, input_length, first_pred_feature)

  def _get_split_list_from_tune_in_idx(self, ratio, seed):
      shuffled_tune_names = list(self.tune_in_idx.keys())
      song_names_without_version = [re.sub(r"-v\d+$", "", tune) for tune in shuffled_tune_names]
      song_dict = {}
      for song, orig_song in zip(song_names_without_version, shuffled_tune_names):
        if song not in song_dict:
          song_dict[song] = []
        song_dict[song].append(orig_song)
      unique_song_names = list(song_dict.keys())
      random.seed(seed)
      random.shuffle(unique_song_names)
      num_train = int(len(unique_song_names)*ratio)
      num_valid = int(len(unique_song_names)*(1-ratio)/2)
      train_names = []
      valid_names = []
      test_names = []
      for song_name in unique_song_names[:num_train]:
        train_names.extend(song_dict[song_name])
      for song_name in unique_song_names[num_train:num_train+num_valid]:
        valid_names.extend(song_dict[song_name])
      for song_name in unique_song_names[num_train+num_valid:]:
        test_names.extend(song_dict[song_name])
      return train_names, valid_names, test_names, shuffled_tune_names