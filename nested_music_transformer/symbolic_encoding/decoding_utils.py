import os, sys
from pathlib import Path

import matplotlib.pyplot as plt
from collections import defaultdict

from music21 import converter
import muspy
import miditoolkit
from miditoolkit.midi.containers import Marker, Instrument, TempoChange, Note, TimeSignature

from .midi2audio import FluidSynth
from data_representation.constants import PROGRAM_INSTRUMENT_MAP

class MuteWarn:
  def __enter__(self):
    self._init_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")

  def __exit__(self, exc_type, exc_val, exc_tb):
    sys.stdout.close()
    sys.stdout = self._init_stdout

def save_score_image_from_midi(midi_fn, file_name):
  assert isinstance(midi_fn, str)
  with MuteWarn():
    convert = converter.parse(midi_fn)
    convert.write('musicxml.png', fp=file_name)

def save_pianoroll_image_from_midi(midi_fn, file_name):
  assert isinstance(midi_fn, str)
  midi_obj_muspy = muspy.read_midi(midi_fn)
  midi_obj_muspy.show_pianoroll(track_label='program', preset='frame')
  plt.gcf().set_size_inches(20, 10)
  plt.savefig(file_name)
  plt.close()

def save_wav_from_midi(midi_fn, file_name, qpm=80):
  assert isinstance(midi_fn, str)
  with MuteWarn():
    music = muspy.read_midi(midi_fn)
    music.tempos[0].qpm = qpm
    music.write_audio(file_name, rate=44100, gain=3)

def save_wav_from_midi_fluidsynth(midi_fn, file_name, gain=3):
  assert isinstance(midi_fn, str)
  fs = FluidSynth(gain=gain)
  fs.midi_to_audio(midi_fn, file_name)

class MidiDecoder4REMI:
  def __init__(
      self, 
      vocab, 
      in_beat_resolution, 
      dataset_name
  ):
    self.vocab = vocab
    self.in_beat_resolution = in_beat_resolution
    self.dataset_name = dataset_name
    if dataset_name == 'SymphonyMIDI':
      self.gain = 0.7
    elif dataset_name == 'SOD' or dataset_name == 'LakhClean':
      self.gain = 1.1
    elif dataset_name == 'Pop1k7' or dataset_name == 'Pop909':
      self.gain = 2.5
    else:
      self.gain = 1.5

  def __call__(self, generated_output, output_path=None):
    '''
    generated_output: list of tensor, the tensor
    '''
    idx2event = self.vocab.idx2event
    if generated_output.dim() == 2:
      generated_output = generated_output.squeeze(0)
    events = [idx2event[token.item()] for token in generated_output]

    midi_obj = miditoolkit.midi.parser.MidiFile()
    if 'tempo' not in idx2event.keys():
      default_tempo = 95
      midi_obj.tempo_changes.append(
        TempoChange(tempo=default_tempo, time=0))
    default_ticks_per_beat = 480
    default_in_beat_ticks  =  480 // self.in_beat_resolution
    cur_pos = 0
    bar_pos = 0
    cur_bar_resol = 0
    beat_pos = 0
    cur_instr = 0 if not self.dataset_name == 'BachChorale' else 53
    instr_notes_dict = defaultdict(list)
    for i in range(len(events)-2):
      cur_event = events[i]
      # print(cur_event)
      name = cur_event.split('_')[0]
      attr = cur_event.split('_')
      if name == 'Bar':
        bar_pos += cur_bar_resol
        if 'time' in cur_event:
          cur_num, cur_denom = attr[-1].split('/')
          new_bar_resol = int(default_ticks_per_beat * int(cur_num) * (4 / int(cur_denom)))
          cur_bar_resol = new_bar_resol
          midi_obj.time_signature_changes.append(
            TimeSignature(numerator=int(cur_num), denominator=int(cur_denom), time=bar_pos))
      elif name == 'Beat':
        beat_pos = int(attr[1])
        cur_pos = bar_pos + beat_pos * default_in_beat_ticks
      elif name == 'Chord':
        chord_text = attr[1] + '_' + attr[2]
        midi_obj.markers.append(Marker(text=chord_text, time=cur_pos))
      elif name == 'Tempo':
        midi_obj.tempo_changes.append(
            TempoChange(tempo=int(attr[1]), time=cur_pos))
      elif name == 'Instrument':
        cur_instr = int(attr[1])
      else:
        if len(self.vocab.feature_list) == 7 or len(self.vocab.feature_list) == 8:
          if 'Note_Pitch' in events[i] and \
          'Note_Duration' in events[i+1] and \
          'Note_Velocity' in events[i+2]:
            pitch = int(events[i].split('_')[-1])
            duration = int(events[i+1].split('_')[-1])
            duration = duration * default_in_beat_ticks
            end = cur_pos + duration 
            velocity = int(events[i+2].split('_')[-1])
            instr_notes_dict[cur_instr].append(
              Note(
                pitch=pitch, 
                start=cur_pos, 
                end=end, 
                velocity=velocity))
        elif len(self.vocab.feature_list) == 4 or len(self.vocab.feature_list) == 5:
          if 'Note_Pitch' in events[i] and \
          'Note_Duration' in events[i+1]:
            pitch = int(events[i].split('_')[-1])
            duration = int(events[i+1].split('_')[-1])
            duration = duration * default_in_beat_ticks
            end = cur_pos + duration 
            velocity = 90
            instr_notes_dict[cur_instr].append(
              Note(
                pitch=pitch, 
                start=cur_pos, 
                end=end, 
                velocity=velocity))
          
    # save midi  
    for instr, notes in instr_notes_dict.items():
      instrument_name = PROGRAM_INSTRUMENT_MAP[instr]
      if instr == 114: is_drum = True
      else: is_drum = False
      instr_track = Instrument(instr, is_drum=is_drum, name=instrument_name)
      instr_track.notes = notes
      midi_obj.instruments.append(instr_track)

    if isinstance(output_path, str) or isinstance(output_path, Path):
      output_path = str(output_path)
      midi_obj.dump(output_path)
      save_pianoroll_image_from_midi(output_path, output_path.replace('.mid', '.png'))
      save_wav_from_midi_fluidsynth(output_path, output_path.replace('.mid', '.wav'), gain=self.gain)
    return midi_obj

class MidiDecoder4CP(MidiDecoder4REMI):
  def __init__(self, vocab, in_beat_resolution, dataset_name):
    super().__init__(vocab, in_beat_resolution, dataset_name)
  
  def _update_chord_tempo(self, midi_obj, cur_pos, token_with_7infos, feature2idx):
    if len(feature2idx) == 7 or len(feature2idx) == 8:
      # chord
      if token_with_7infos[feature2idx['chord']] != 'CONTI' and token_with_7infos[feature2idx['chord']] != 0:
        midi_obj.markers.append(
          Marker(text=str(token_with_7infos[feature2idx['chord']]), time=cur_pos))
      # tempo
      if token_with_7infos[feature2idx['tempo']] != 'CONTI' and token_with_7infos[feature2idx['tempo']] != 0 and token_with_7infos[feature2idx['tempo']] != "Tempo_N_N":
        tempo = int(token_with_7infos[feature2idx['tempo']].split('_')[-1])
        midi_obj.tempo_changes.append(
          TempoChange(tempo=tempo, time=cur_pos))
      return midi_obj
    elif len(feature2idx) == 4 or len(feature2idx) == 5:
      return midi_obj
    
  def __call__(self, generated_output, output_path=None):
    '''
    generated_output: tensor, batch x seq_len x num_types
    num_types includes: type, tempo, chord,'beat, pitch, duration, velocity
    '''
    idx2event = self.vocab.idx2event
    feature_keys = self.vocab.feature_list
    feature2idx = {key: idx for idx, key in enumerate(feature_keys)}

    midi_obj = miditoolkit.midi.parser.MidiFile()
    if len(feature2idx) == 4 or len(feature2idx) == 5:
      default_tempo = 95
      midi_obj.tempo_changes.append(
        TempoChange(tempo=default_tempo, time=0))
    default_ticks_per_beat = 480
    default_in_beat_ticks  =  480 // self.in_beat_resolution
    cur_pos = 0
    bar_pos = 0
    cur_bar_resol = 0
    beat_pos = 0
    instr_notes_dict = defaultdict(list)
    generated_output = generated_output.squeeze(0)
    for i in range(len(generated_output)):
      token_with_7infos = []
      for kidx, key in enumerate(feature_keys):
        token_with_7infos.append(idx2event[key][generated_output[i][kidx].item()])
      # type token
      if 'time_signature' in token_with_7infos[feature2idx['type']]:
        cur_num, cur_denom = token_with_7infos[feature2idx['type']].split('_')[-1].split('/')
        bar_pos += cur_bar_resol
        new_bar_resol = int(default_ticks_per_beat * int(cur_num) * (4 / int(cur_denom)))
        cur_bar_resol = new_bar_resol
        midi_obj.time_signature_changes.append(
          TimeSignature(numerator=int(cur_num), denominator=int(cur_denom), time=bar_pos))
      elif token_with_7infos[feature2idx['type']] == 'Metrical':
        if token_with_7infos[feature2idx['beat']] == 'Bar':
          bar_pos += cur_bar_resol
        elif 'Beat' in str(token_with_7infos[feature2idx['beat']]):
          beat_pos = int(token_with_7infos[feature2idx['beat']].split('_')[1])
          cur_pos = bar_pos + beat_pos * default_in_beat_ticks # ticks
          # chord and tempo
          midi_obj = self._update_chord_tempo(midi_obj, cur_pos, token_with_7infos, feature2idx)
      elif token_with_7infos[feature2idx['type']] == 'Note':
        # instrument token
        if len(feature2idx) == 8 or len(feature2idx) == 5:
          if token_with_7infos[feature2idx['instrument']] != 0 and token_with_7infos[feature2idx['instrument']] != 'CONTI':
            cur_instr = int(token_with_7infos[feature2idx['instrument']].split('_')[-1])
        else:
          cur_instr = 0 if not self.dataset_name == 'BachChorale' else 53
        try:
          pitch = token_with_7infos[feature2idx['pitch']].split('_')[-1]
          duration = token_with_7infos[feature2idx['duration']].split('_')[-1]
          duration = int(duration) * default_in_beat_ticks
          if len(feature2idx) == 7 or len(feature2idx) == 8:
            velocity = token_with_7infos[feature2idx['velocity']].split('_')[-1]
          else:
            velocity = 80
          end = cur_pos + duration
          instr_notes_dict[cur_instr].append(
            Note(
              pitch=int(pitch), 
              start=cur_pos, 
              end=end, 
              velocity=int(velocity))
            )
        except:
          continue
      else: # when new bar started without beat
        continue

    # save midi
    for instr, notes in instr_notes_dict.items():
      instrument_name = PROGRAM_INSTRUMENT_MAP[instr]
      if instr == 114: is_drum = True
      else: is_drum = False
      instr_track = Instrument(instr, is_drum=is_drum, name=instrument_name)
      instr_track.notes = notes
      midi_obj.instruments.append(instr_track)

    if isinstance(output_path, str) or isinstance(output_path, Path):
      output_path = str(output_path)
      midi_obj.dump(output_path)
      save_pianoroll_image_from_midi(output_path, output_path.replace('.mid', '.png'))
      save_wav_from_midi_fluidsynth(output_path, output_path.replace('.mid', '.wav'), gain=self.gain)
    return midi_obj
  
class MidiDecoder4NB(MidiDecoder4REMI):
  def __init__(self, vocab, in_beat_resolution, dataset_name):
    super().__init__(vocab, in_beat_resolution, dataset_name)
  
  def _update_additional_info(self, midi_obj, cur_pos, token_with_7infos, feature2idx):
    if len(feature2idx) == 7 or len(feature2idx) == 8:
      # chord
      if token_with_7infos[feature2idx['chord']] != 'CONTI' and token_with_7infos[feature2idx['chord']] != 0 and token_with_7infos[feature2idx['chord']] != 'Chord_N_N':
        midi_obj.markers.append(
          Marker(text=str(token_with_7infos[feature2idx['chord']]), time=cur_pos))
      # tempo
      if token_with_7infos[feature2idx['tempo']] != 'CONTI' and token_with_7infos[feature2idx['tempo']] != 0 and token_with_7infos[feature2idx['tempo']] != "Tempo_N_N":
        tempo = int(token_with_7infos[feature2idx['tempo']].split('_')[-1])
        midi_obj.tempo_changes.append(
          TempoChange(tempo=tempo, time=cur_pos))
      return midi_obj
    elif len(feature2idx) == 4 or len(feature2idx) == 5:
      return midi_obj
    
  def __call__(self, generated_output, output_path=None):
    '''
    generated_output: tensor, batch x seq_len x num_types
    num_types includes: type, beat, chord, tempo, intrument, pitch, duration, velocity
    '''
    idx2event = self.vocab.idx2event
    feature_keys = self.vocab.feature_list
    feature2idx = {key: idx for idx, key in enumerate(feature_keys)}

    midi_obj = miditoolkit.midi.parser.MidiFile()
    if len(feature2idx) == 4 or len(feature2idx) == 5:
      default_tempo = 95
      midi_obj.tempo_changes.append(
        TempoChange(tempo=default_tempo, time=0))
    default_ticks_per_beat = 480
    default_in_beat_ticks = 480 // self.in_beat_resolution
    cur_pos = 0
    bar_pos = 0
    cur_bar_resol = 0
    beat_pos = 0
    instr_notes_dict = defaultdict(list)
    generated_output = generated_output.squeeze(0)
    for i in range(len(generated_output)):
      token_with_7infos = []
      for kidx, key in enumerate(feature_keys):
        token_with_7infos.append(idx2event[key][generated_output[i][kidx].item()])
      # type token
      if token_with_7infos[feature2idx['type']] == 'Empty_Bar' or token_with_7infos[feature2idx['type']] == 'SNN':
        bar_pos += cur_bar_resol
      elif 'NNN' in token_with_7infos[feature2idx['type']]:
        cur_num, cur_denom = token_with_7infos[feature2idx['type']].split('_')[-1].split('/')
        bar_pos += cur_bar_resol
        new_bar_resol = int(default_ticks_per_beat * int(cur_num) * (4 / int(cur_denom)))
        cur_bar_resol = new_bar_resol
        midi_obj.time_signature_changes.append(
          TimeSignature(numerator=int(cur_num), denominator=int(cur_denom), time=bar_pos))
      # instrument token
      if len(feature2idx) == 8 or len(feature2idx) == 5:
        if token_with_7infos[feature2idx['instrument']] != 0 and token_with_7infos[feature2idx['instrument']] != 'CONTI':
          cur_instr = int(token_with_7infos[feature2idx['instrument']].split('_')[-1])
      else:
        cur_instr = 0 if not self.dataset_name == 'BachChorale' else 53
      if 'Beat' in str(token_with_7infos[feature2idx['beat']]) or 'CONTI' in str(token_with_7infos[feature2idx['beat']]):
        if 'Beat' in str(token_with_7infos[feature2idx['beat']]): # when beat is not CONTI beat is updated
          beat_pos = int(token_with_7infos[feature2idx['beat']].split('_')[1])
          cur_pos = bar_pos + beat_pos * default_in_beat_ticks # ticks
        # update chord and tempo
        midi_obj = self._update_additional_info(midi_obj, cur_pos, token_with_7infos, feature2idx)  
        # note
        try:
          pitch = token_with_7infos[feature2idx['pitch']].split('_')[-1]
          duration = token_with_7infos[feature2idx['duration']].split('_')[-1] # duration between 1~192
          duration = int(duration) * default_in_beat_ticks
          if len(feature2idx) == 7 or len(feature2idx) == 8:
            velocity = token_with_7infos[feature2idx['velocity']].split('_')[-1]
          else:
            velocity = 90
          end = cur_pos + duration
          instr_notes_dict[cur_instr].append(
            Note(
              pitch=int(pitch), 
              start=cur_pos, 
              end=end, 
              velocity=int(velocity))
            )
        except:
          continue
      else: # when new bar started without beat
        continue
    
    # save midi
    for instr, notes in instr_notes_dict.items():
      instrument_name = PROGRAM_INSTRUMENT_MAP[instr]
      if instr == 114: is_drum = True
      else: is_drum = False
      instr_track = Instrument(instr, is_drum=is_drum, name=instrument_name)
      instr_track.notes = notes
      midi_obj.instruments.append(instr_track)

    if isinstance(output_path, str) or isinstance(output_path, Path):
      output_path = str(output_path)
      midi_obj.dump(output_path)
      save_pianoroll_image_from_midi(output_path, output_path.replace('.mid', '.png'))
      save_wav_from_midi_fluidsynth(output_path, output_path.replace('.mid', '.wav'), gain=self.gain)
    return midi_obj
