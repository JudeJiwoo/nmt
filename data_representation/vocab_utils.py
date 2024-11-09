import pickle
from pathlib import Path
from typing import Union
from multiprocessing import Pool, cpu_count
from collections import defaultdict
from fractions import Fraction

import torch

import json
from tqdm import tqdm

def sort_key(s):
  fraction_part = s.split('_')[-1]
  numerator, denominator = map(int, fraction_part.split('/'))
  # Return a tuple with denominator first, then numerator, both in negative for descending order
  return (-denominator, -numerator)

class LangTokenVocab:
  def __init__(
    self, 
    in_vocab_file_path:Union[Path, None],
    event_data: list, 
    encoding_scheme: str, 
    num_features: int
  ):
    '''
    Initializes the LangTokenVocab class.
    
    Args:
      in_vocab_file_path (Union[Path, None]): Path to the pre-made vocabulary file (optional).
      event_data (list): List of event data to create a vocabulary if no pre-made vocab is provided.
      encoding_scheme (str): Encoding scheme to be used (e.g., 'remi', 'cp', 'nb').
      num_features (int): Number of features to be used (e.g., 4, 5, 7, 8).
    
    Summary:
    This class is responsible for handling vocabularies used in language models, especially for REMI encoding. 
    It supports multiple encoding schemes, creates vocabularies based on event data, handles special tokens (e.g., 
    start/end of sequence), and manages feature-specific masks. It provides methods for saving, loading, and decoding
    vocabularies. It also supports vocabulary augmentation for pitch, instrument, beat, and chord features, ensuring 
    that these are arranged and ordered appropriately.

    For all encoding schemes, the metric or special tokens are named as 'type', 
    so that we can easily handle and compare among different encoding schemes.
    '''

    self.encoding_scheme = encoding_scheme
    self.num_features = num_features
    self._prepare_in_vocab(in_vocab_file_path, event_data)  # Prepares initial vocab based on the input file or event data
    self._get_features()  # Extracts relevant features based on the num_features
    self.idx2event, self.event2idx = self._get_vocab(event_data, unique_vocabs=self.idx2event)  # Creates vocab or loads premade vocab
    if self.encoding_scheme == 'remi':
      self._make_mask()  # Generates masks for 'remi' encoding scheme
    self._get_sos_eos_token()  # Retrieves special tokens (Start of Sequence, End of Sequence)

  # Prepares vocabulary if a pre-made vocab file exists or handles cases with no input file.
  def _prepare_in_vocab(self, in_vocab_file_path, event_data):
    if in_vocab_file_path is not None and in_vocab_file_path.exists():
      with open(in_vocab_file_path, 'r') as f:
        idx2event_temp = json.load(f)
      if self.encoding_scheme == 'cp' or self.encoding_scheme == 'nb':
        for key in idx2event_temp.keys():
          idx2event_temp[key] = {int(idx):tok for idx, tok in idx2event_temp[key].items()}
      elif self.encoding_scheme == 'remi':
        idx2event_temp = {int(idx):tok for idx, tok in idx2event_temp.items()}
      self.idx2event = idx2event_temp
    elif in_vocab_file_path is None and event_data is None:
      raise NotImplementedError('either premade vocab or event_data should be given')
    else:
      self.idx2event = None

  # Extracts features depending on the number of features chosen (4, 5, 7, 8).
  def _get_features(self):
    feature_args = {
      4: ["type", "beat", "pitch", "duration"],
      5: ["type", "beat", "instrument", "pitch", "duration"],
      7: ["type", "beat", "chord", "tempo", "pitch", "duration", "velocity"],
      8: ["type", "beat", "chord", "tempo", "instrument", "pitch", "duration", "velocity"]}
    self.feature_list = feature_args[self.num_features]

  # Saves the current vocabulary to a specified JSON path.
  def save_vocab(self, json_path):
    with open(json_path, 'w') as f:
      json.dump(self.idx2event, f, indent=2, ensure_ascii=False)

  # Returns the size of the current vocabulary.
  def get_vocab_size(self):
    return len(self.idx2event)

  # Handles Start of Sequence (SOS) and End of Sequence (EOS) tokens based on the encoding scheme.
  def _get_sos_eos_token(self):
    if self.encoding_scheme == 'remi':
      self.sos_token = [self.event2idx['SOS_None']]
      self.eos_token = [[self.event2idx['EOS_None']]]
    else:
      self.sos_token = [[self.event2idx['type']['SOS']] + [0] * (self.num_features - 1)]
      self.eos_token = [[self.event2idx['type']['EOS']] + [0] * (self.num_features - 1)]

  # Generates vocabularies by either loading from a file or creating them based on the event data.
  def _get_vocab(self, event_data, unique_vocabs=None):
    # make new vocab from given event_data
    if event_data is not None:
      unique_char_list = list(set([f'{event["name"]}_{event["value"]}' for tune_path in event_data for event in pickle.load(open(tune_path, 'rb'))]))
      unique_vocabs = sorted(unique_char_list)
      unique_vocabs.remove('SOS_None')
      unique_vocabs.remove('EOS_None')
      unique_vocabs.remove('Bar_None')
      new_unique_vocab = self._augment_pitch_vocab(unique_vocabs)
      if self.num_features == 5 or self.num_features == 8:
        new_unique_vocab = self._arange_instrument_vocab(new_unique_vocab)
      if self.num_features == 7 or self.num_features == 8:
        new_unique_vocab = self._arange_chord_vocab(new_unique_vocab)
      new_unique_vocab = self._arange_beat_vocab(new_unique_vocab)
      new_unique_vocab.insert(0, 'SOS_None')
      new_unique_vocab.insert(1, 'EOS_None')
      new_unique_vocab.insert(2, 'Bar_None')
      idx2event = {int(idx) : tok for idx, tok in enumerate(new_unique_vocab)}
      event2idx = {tok : int(idx) for idx, tok in idx2event.items()}
    # load premade vocab
    else:
      idx2event = unique_vocabs
      event2idx = {tok : int(idx) for idx, tok in unique_vocabs.items()}
    return idx2event, event2idx

  # Augments the pitch vocabulary by expanding the range of pitch values.
  def _augment_pitch_vocab(self, unique_vocabs):
    pitch_vocab = [x for x in unique_vocabs if 'Note_Pitch_' in x]
    pitch_int = [int(x.replace('Note_Pitch_', '')) for x in pitch_vocab if x.replace('Note_Pitch_', '').isdigit()]
    min_pitch = min(pitch_int)
    max_pitch = max(pitch_int)
    min_pitch_margin = max(min_pitch-6, 0)
    max_pitch_margin = min(max_pitch+7, 127)
    new_pitch_vocab = sorted([f'Note_Pitch_{x}' for x in range(min_pitch_margin, max_pitch_margin+1)], key=lambda x: (not isinstance(x, int), int(x.split('_')[-1] if isinstance(x, str) else x)))
    new_unique_vocab = [x for x in unique_vocabs if x not in new_pitch_vocab] + new_pitch_vocab
    return new_unique_vocab

  # Orders and arranges the instrument vocabulary.
  def _arange_instrument_vocab(self, unique_vocabs):
    instrument_vocab = [x for x in unique_vocabs if 'Instrument_' in x]
    new_instrument_vocab = sorted(instrument_vocab, key=lambda x: (not isinstance(x, int), int(x.split('_')[-1] if isinstance(x, str) else x)))
    new_unique_vocab = [x for x in unique_vocabs if x not in new_instrument_vocab] + new_instrument_vocab
    return new_unique_vocab

  # Orders and arranges the chord vocabulary, ensuring 'Chord_N_N' is the last token.
  def _arange_chord_vocab(self, unique_vocabs):
    '''
    for chord augmentation
    Chord_N_N should be the last token in the list for an easy implementation of chord augmentation
    '''
    chord_vocab = [x for x in unique_vocabs if 'Chord_' in x]
    chord_vocab.remove('Chord_N_N')
    new_chord_vocab = sorted(chord_vocab, key=lambda x: (not isinstance(x, int), x.split('_')[-1] if isinstance(x, str) else x, x.split('_')[1] if isinstance(x, str) else x))
    new_chord_vocab.append('Chord_N_N')
    new_unique_vocab = [x for x in unique_vocabs if x not in new_chord_vocab] + new_chord_vocab
    return new_unique_vocab

  # Orders and arranges the beat vocabulary.
  def _arange_beat_vocab(self, unique_vocabs):
    beat_vocab = [x for x in unique_vocabs if 'Beat_' in x]
    new_beat_vocab = sorted(beat_vocab, key=lambda x: (not isinstance(x, int), int(x.split('_')[-1] if isinstance(x, str) else x)))
    count = 0
    for idx, token in enumerate(unique_vocabs):
      if 'Beat_' in token:
        unique_vocabs[idx] = new_beat_vocab[count]
        count += 1
    return unique_vocabs     

  # Generates masks for the 'remi' encoding scheme.
  def _make_mask(self):
    '''
    This function is used to extract the target musical features for validation.
    '''
    idx2feature = {}
    for idx, feature in self.idx2event.items():
      if feature.startswith('SOS') or feature.startswith('EOS') or feature.startswith('Bar'):
        idx2feature[idx] = 'type'
      elif feature.startswith('Beat'):
        idx2feature[idx] = 'beat'
      elif feature.startswith('Chord'):
        idx2feature[idx] = 'chord'
      elif feature.startswith('Tempo'):
        idx2feature[idx] = 'tempo'
      elif feature.startswith('Note_Pitch'):
        idx2feature[idx] = 'pitch'
      elif feature.startswith('Note_Duration'):
        idx2feature[idx] = 'duration'
      elif feature.startswith('Note_Velocity'):
        idx2feature[idx] = 'velocity'
      elif feature.startswith('Instrument'):
        idx2feature[idx] = 'instrument'

    self.total_mask = {}
    self.remi_vocab_boundaries_by_key = {}
    for target in self.feature_list:
      mask = [0] * len(idx2feature)  # Initialize all-zero list of length equal to dictionary
      for key, value in idx2feature.items():
        if value == target:
          mask[int(key)] = 1  # If value equals target, set corresponding position in mask to 1
      mask = torch.LongTensor(mask)
      self.total_mask[target] = mask
      start_idx, end_idx = torch.argwhere(mask == 1).flatten().tolist()[0], torch.argwhere(mask == 1).flatten().tolist()[-1]
      self.remi_vocab_boundaries_by_key[target] = (start_idx, end_idx+1)

  def decode(self, events:torch.Tensor):
    '''
    Used for checking events in the evaluation
    events: 1d tensor
    '''
    decoded_list = []
    for event in events:
      decoded_list.append(self.idx2event[event.item()])
    return decoded_list

  def __call__(self, word):
    '''
    for remi style encoding
    '''
    return self.event2idx[f"{word['name']}_{word['value']}"]

class MusicTokenVocabCP(LangTokenVocab):
  def __init__(
    self, 
    in_vocab_file_path:Union[Path, None],
    event_data: list, 
    encoding_scheme: str, 
    num_features: int
  ):
    # Initialize the vocabulary class with vocab file path, event data, encoding scheme, and feature count
    super().__init__(in_vocab_file_path, event_data, encoding_scheme, num_features)

  def _augment_pitch_vocab(self, unique_vocabs):
    # Extract pitch-related vocabularies and adjust pitch range
    pitch_total_vocab = unique_vocabs['pitch']
    pitch_vocab = [x for x in pitch_total_vocab if 'Note_Pitch_' in str(x)]
    pitch_int = [int(x.replace('Note_Pitch_', '')) for x in pitch_vocab if x.replace('Note_Pitch_', '').isdigit()]
    # Determine the min and max pitch values and extend the pitch range slightly
    min_pitch = min(pitch_int)
    max_pitch = max(pitch_int)
    min_pitch_margin = max(min_pitch - 6, 0)
    max_pitch_margin = min(max_pitch + 7, 127)
    # Create new pitch vocab and ensure new entries do not overlap with existing ones
    new_pitch_vocab = [f'Note_Pitch_{x}' for x in range(min_pitch_margin, max_pitch_margin + 1)]
    new_pitch_vocab = [x for x in pitch_total_vocab if str(x) not in new_pitch_vocab] + new_pitch_vocab
    unique_vocabs['pitch'] = new_pitch_vocab
    return unique_vocabs

  def _mp_get_unique_vocab(self, tune, features):
    # Read event data from a file and collect unique vocabularies for specified features
    with open(tune, 'rb') as f:
        events_list = pickle.load(f)
    unique_vocabs = defaultdict(set)
    for event in events_list:
        for key in features:
            unique_vocabs[key].add(event[key])
    return unique_vocabs

  def _get_chord_vocab(self):
    '''
    Manually define the chord vocabulary by combining roots and qualities
    from a predefined list. This is used for chord augmentation.
    '''
    root_list = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']
    quality_list = ['+', '/o7', '7', 'M', 'M7', 'm', 'm7', 'o', 'o7', 'sus2', 'sus4']
    chord_vocab = [f'Chord_{root}_{quality}' for root in root_list for quality in quality_list]
    # Sort the chord vocabulary based on the root and quality
    chord_vocab = sorted(chord_vocab, key=lambda x: (not isinstance(x, int), x.split('_')[-1] if isinstance(x, str) else x, x.split('_')[0] if isinstance(x, str) else x))
    return chord_vocab

  def _cp_sort_type(self, unique_vocabs):
    # Similar to _nb_sort_type but used for the 'cp' encoding scheme, sorting vocabularies in a different order
    unique_vocabs.remove('SOS')
    unique_vocabs.remove('EOS')
    unique_vocabs.remove('Metrical')
    unique_vocabs.remove('Note')
    vocab_list = list(unique_vocabs)
    unique_vocabs = sorted(vocab_list, key=sort_key)
    unique_vocabs.insert(0, 'SOS')
    unique_vocabs.insert(1, 'EOS')
    unique_vocabs.insert(2, 'Metrical')
    unique_vocabs.insert(3, 'Note')
    return unique_vocabs

  # Define custom sorting function
  def sort_type_cp(self, item):
      if item == 0:
          return (0, 0)  # Move 0 to the beginning
      elif isinstance(item, str):
          if item.startswith("Bar"):
              return (1, item)  # "Bar" items come next, sorted lexicographically
          elif item.startswith("Beat"):
              # Extract numeric part of "Beat_x" to sort numerically
              beat_number = int(item.split('_')[1])
              return (2, beat_number)  # "Beat" items come last, sorted by number
      return (3, item)  # Catch-all for anything unexpected (shouldn't be necessary here)

  def _get_vocab(self, event_data, unique_vocabs=None):
    if event_data is not None:
        # Create vocab mappings (event2idx, idx2event) from the provided event data
        print('start to get unique vocab')
        event2idx = {}
        idx2event = {}
        unique_vocabs = defaultdict(set)
        # Use multiprocessing to extract unique vocabularies for each event
        with Pool(cpu_count()) as p:
            results = p.starmap(self._mp_get_unique_vocab, tqdm([(tune, self.feature_list) for tune in event_data]))
        # Combine results from different processes
        for result in results:
            for key in self.feature_list:
                if key == 'chord':  # Chords are handled separately
                    continue
                unique_vocabs[key].update(result[key])
        # Augment pitch vocab and add manually defined chord vocab
        unique_vocabs = self._augment_pitch_vocab(unique_vocabs)
        unique_vocabs['chord'] = self._get_chord_vocab()
        # Process each feature type, handling special cases like 'tempo' and 'chord'
        for key in self.feature_list:
            if key == 'tempo':
                remove_nn_flag = False
                if 'Tempo_N_N' in unique_vocabs[key]:
                    unique_vocabs[key].remove('Tempo_N_N')
                    remove_nn_flag = True
                unique_vocabs[key] = sorted(unique_vocabs[key], key=lambda x: (not isinstance(x, int), int(x.split('_')[-1] if isinstance(x, str) else x)))
                if remove_nn_flag:
                    unique_vocabs[key].insert(1, 'Tempo_N_N')
            elif key == 'chord':
                unique_vocabs[key].insert(0, 0)
                unique_vocabs[key].insert(1, 'Chord_N_N')
            elif key == 'type':  # Sort 'type' vocab depending on the encoding scheme
                if self.encoding_scheme == 'cp':
                    unique_vocabs[key] = self._cp_sort_type(unique_vocabs[key])
                else:  # NB encoding scheme
                    unique_vocabs[key] = self._nb_sort_type(unique_vocabs[key])
            elif key == 'beat' and self.encoding_scheme == 'cp':  # Handle 'beat' vocab with 'cp' scheme
                # unique_vocabs[key].remove('Bar')
                # unique_vocabs[key] = sorted(unique_vocabs[key], key=lambda x: (not isinstance(x, int), Fraction(x.split('_')[-1] if isinstance(x, str) else x)))
                # unique_vocabs[key].insert(1, 'Bar')
                unique_vocabs[key] = sorted(unique_vocabs[key], key = self.sort_type_cp)
            elif key == 'beat' and self.encoding_scheme == 'nb':  # Handle 'beat' vocab with 'nb' scheme
                unique_vocabs[key] = sorted(unique_vocabs[key], key=lambda x: (not isinstance(x, int), int(x.split('_')[-1] if isinstance(x, str) else x)))
            elif key == 'instrument':  # Sort 'instrument' vocab by integer values
                unique_vocabs[key] = sorted(unique_vocabs[key], key=lambda x: (not isinstance(x, int), int(x.split('_')[-1] if isinstance(x, str) else x)))
            else:  # Default case: sort by integer values for other keys
                unique_vocabs[key] = sorted(unique_vocabs[key], key=lambda x: (not isinstance(x, int), int(x.split('_')[-1] if isinstance(x, str) else x)))
            # Create event2idx and idx2event mappings for each feature
            event2idx[key] = {tok: int(idx) for idx, tok in enumerate(unique_vocabs[key])}
            idx2event[key] = {int(idx): tok for idx, tok in enumerate(unique_vocabs[key])}
        return idx2event, event2idx
    else:
        # If no event data, simply map unique vocab to indexes
        event2idx = {}
        for key in self.feature_list:
            event2idx[key] = {tok: int(idx) for idx, tok in unique_vocabs[key].items()}
        return unique_vocabs, event2idx

  def get_vocab_size(self):
    # Return the size of the vocabulary for each feature
    return {key: len(self.idx2event[key]) for key in self.feature_list}

  def __call__(self, event):
    # Convert an event to its corresponding indices
    return [self.event2idx[key][event[key]] for key in self.feature_list]
        
  def decode(self, events:torch.Tensor):
    decoded_list = []
    for event in events:
      decoded_list.append([self.idx2event[key][event[idx].item()] for idx, key in enumerate(self.feature_list)])
    return decoded_list

class MusicTokenVocabNB(MusicTokenVocabCP):
  def __init__(
    self, 
    in_vocab_file_path:Union[Path, None],
    event_data: list, 
    encoding_scheme: str, 
    num_features: int
  ):
    super().__init__(in_vocab_file_path, event_data, encoding_scheme, num_features)

  def _nb_sort_type(self, unique_vocabs):
    # Remove special tokens and sort the remaining vocab list, then re-insert the special tokens in order
    unique_vocabs.remove('SOS')
    unique_vocabs.remove('EOS')
    unique_vocabs.remove('Empty_Bar')
    unique_vocabs.remove('SSS')
    unique_vocabs.remove('SSN')
    unique_vocabs.remove('SNN')
    vocab_list = list(unique_vocabs)
    unique_vocabs = sorted(vocab_list, key=sort_key)
    unique_vocabs.insert(0, 'SOS')
    unique_vocabs.insert(1, 'EOS')
    unique_vocabs.insert(2, 'Empty_Bar')
    unique_vocabs.insert(3, 'SSS')
    unique_vocabs.insert(4, 'SSN')
    unique_vocabs.insert(5, 'SNN')
    return unique_vocabs