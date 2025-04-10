import argparse
import time
import itertools
import copy
from copy import deepcopy
from pathlib import Path
from multiprocessing import Pool, cpu_count
from collections import defaultdict
from fractions import Fraction
from typing import List

import numpy as np
import pickle
from tqdm import tqdm

import miditoolkit
from miditoolkit.midi.containers import Marker, Instrument
from chorder import Dechorder

from constants import NUM2PITCH, PROGRAM_INSTRUMENT_MAP, INSTRUMENT_PROGRAM_MAP

'''
This script is designed to preprocess MIDI files and convert them into a structured corpus suitable for symbolic music analysis or model training. 
It handles various tasks, including setting beat resolution, calculating duration, velocity, and tempo bins, and processing MIDI data into quantized musical events. 
'''

def get_tempo_bin(max_tempo:int, ratio:float=1.1):
  bpm = 30
  regular_tempo_bins = [bpm]
  while bpm < max_tempo:
    bpm *= ratio
    bpm = round(bpm)
    if bpm > max_tempo:
      break
    regular_tempo_bins.append(bpm)
  return np.array(regular_tempo_bins)

def split_markers(markers:List[miditoolkit.midi.containers.Marker]):
  '''
  split markers into chord, tempo, label
  '''
  chords = []
  for marker in markers:
    splitted_text = marker.text.split('_')
    if splitted_text[0] != 'global' and 'Boundary' not in splitted_text[0]:
      chords.append(marker)
  return chords

class CorpusMaker():
  def __init__(
      self, 
      dataset_name:str, 
      num_features:int, 
      in_dir:Path, 
      out_dir:Path, 
      debug:bool
  ):
    '''
    Initialize the CorpusMaker with dataset information and directory paths.
    It sets up MIDI paths, output directories, and debug mode, then
    retrieves the beat resolution, duration bins, velocity/tempo bins, and prepares the MIDI file list.
    '''
    self.dataset_name = dataset_name
    self.num_features = num_features
    self.midi_path = in_dir / f"{dataset_name}"
    self.out_dir = out_dir
    self.debug = debug
    self._get_in_beat_resolution()
    self._get_duration_bins()
    self._get_velocity_tempo_bins()
    self._get_min_max_last_time()
    self._prepare_midi_list()
  
  def _get_in_beat_resolution(self):
    # Retrieve the resolution of quarter note based on the dataset name (e.g., 4 means the minimum resolution sets to 16th note)
    in_beat_resolution_dict = {'BachChorale': 4, 'Pop1k7': 4, 'Pop909': 4, 'SOD': 12, 'LakhClean': 4, 'SymphonyMIDI': 8}
    self.in_beat_resolution = in_beat_resolution_dict[self.dataset_name]

  def _get_duration_bins(self):
    # Set up regular duration bins for quantizing note lengths, based on the beat resolution.
    base_duration = {4:[1,2,3,4,5,6,8,10,12,16,20,24,28,32],
                     8:[1,2,3,4,6,8,10,12,14,16,20,24,28,32,36,40,48,56,64],
                     12:[1,2,3,4,6,9,12,15,18,24,30,36,42,48,54,60,72,84,96]}
    base_duration_list = base_duration[self.in_beat_resolution]
    self.regular_duration_bins = np.array(base_duration_list)

  def _get_velocity_tempo_bins(self):
    # Define velocity and tempo bins based on whether the dataset is a performance or score type.
    midi_type_dict = {'BachChorale': 'score', 'Pop1k7': 'perform', 'Pop909': 'score', 'SOD': 'score', 'LakhClean': 'score', 'Symphony': 'score'}
    midi_type = midi_type_dict[self.dataset_name]
    # For performance-type datasets, set finer granularity of velocity and tempo bins.
    if midi_type == 'perform':
      self.regular_velocity_bins = np.array(list(range(40, 128, 8)) + [127])
      self.regular_tempo_bins = get_tempo_bin(max_tempo=240, ratio=1.04)
    # For score-type datasets, use coarser velocity and tempo bins.
    elif midi_type == 'score':
      self.regular_velocity_bins = np.array([40, 60, 80, 100, 120])
      self.regular_tempo_bins = get_tempo_bin(max_tempo=390, ratio=1.1)

  def _get_min_max_last_time(self):
    '''
    Set the minimum and maximum allowed length of a MIDI track, depending on the dataset.
    0 to 2000 means no limitation
    '''
    last_time_dict = {'BachChorale': (0, 2000), 'Pop1k7': (0, 2000), 'Pop909': (0, 2000), 'SOD': (60, 1000), 'LakhClean': (60, 600), 'Symphony': (60, 1500)}
    self.min_last_time, self.max_last_time = last_time_dict[self.dataset_name]

  def _prepare_midi_list(self):
    midi_path = Path(self.midi_path)
    self.midi_list = sorted(list(midi_path.rglob("*.midi")) + list(midi_path.rglob("*.mid")))

  def make_corpus(self) -> None:
    '''
    Main method to process the MIDI files and create the corpus data.
    It supports both single-processing (debug mode) and multi-processing for large datasets.
    '''
    print("preprocessing midi data to corpus data")
    # check the corpus folder is already exist and make it if not
    Path(self.out_dir).mkdir(parents=True, exist_ok=True)
    Path(self.out_dir / f"corpus_{self.dataset_name}").mkdir(parents=True, exist_ok=True)
    Path(self.out_dir / f"midi_{self.dataset_name}").mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    if self.debug:
      # single processing for debugging
      broken_counter = 0
      success_counter = 0
      for file_path in tqdm(self.midi_list, total=len(self.midi_list)):
        message = self._mp_midi2corpus(file_path)
        if message == "error":
          broken_counter += 1
        elif message == "success":
          success_counter += 1
    else:
    # Multi-threaded processing for faster corpus generation.
      broken_counter = 0
      success_counter = 0
      with Pool(cpu_count()) as p:
        for message in tqdm(p.imap(self._mp_midi2corpus, self.midi_list), total=len(self.midi_list)):
          if message == "error":
            broken_counter += 1
          elif message == "success":
            success_counter += 1
    print(f"Making corpus takes: {time.time() - start_time}s, success: {success_counter}, broken: {broken_counter}")

  def _mp_midi2corpus(self, file_path:Path):
    # Converts a single MIDI file to corpus format and saves both the corpus and MIDI.
    try:
      midi_obj = self._analyze(file_path)
      corpus, midi_obj = self._midi2corpus(midi_obj)
      # Save corpus as a pickle file and the corresponding MIDI object.
      filename = file_path.stem + ".pkl"  # Get the stem (filename without extension) of the original file path
      save_path = Path(self.out_dir) / f"corpus_{self.dataset_name}" / filename  # Create a new Path object for saving
      with save_path.open('wb') as f:
        pickle.dump(corpus, f)
      midiname = file_path.stem + ".mid"
      save_path = Path("../dataset/represented_data/corpus") / f"midi_{self.dataset_name}" / midiname
      midi_obj.dump(save_path)
      del midi_obj, corpus
      return "success"
    except (OSError, EOFError, ValueError, KeyError) as e:
      # prin error message
      print(f"Error in {file_path.name}: {e}")
      return "error"
    except AssertionError as e:
      print(f"Error in {file_path.name}: {e}")
      return "error"
    except Exception as e:
      print(f"Error in {file_path.name}: {e}")
      return "error"

  def _check_length(self, last_time:float):
    if last_time < self.min_last_time or last_time > self.max_last_time:
      raise ValueError(f"last time {last_time} is out of range")

  def _analyze(self, midi_path:Path):
    # Loads and analyzes a MIDI file, performing various checks and extracting chords.
    midi_obj = miditoolkit.midi.parser.MidiFile(midi_path)
    
    # check length
    mapping = midi_obj.get_tick_to_time_mapping()
    last_time = mapping[midi_obj.max_tick]
    self._check_length(last_time)
    
    for ins in midi_obj.instruments:
      # delete instrument with no notes
      if len(ins.notes) == 0:
        midi_obj.instruments.remove(ins)
        continue
      notes = ins.notes
      notes = sorted(notes, key=lambda x: (x.start, x.pitch))

    # three steps to merge instruments
    self._merge_percussion(midi_obj)
    self._pruning_instrument(midi_obj)
    self._limit_max_track(midi_obj)

    if self.num_features == 7 or self.num_features == 8:
      # in case of 7 or 8 features, we need to extract chords
      new_midi_obj = self._pruning_notes_for_chord_extraction(midi_obj)
      chords = Dechorder.dechord(new_midi_obj)
      markers = []
      for cidx, chord in enumerate(chords):
        if chord.is_complete():
          chord_text = NUM2PITCH[chord.root_pc] + '_' + chord.quality + '_' + NUM2PITCH[chord.bass_pc]
        else:
          chord_text = 'N_N_N'
        markers.append(Marker(time=int(cidx*new_midi_obj.ticks_per_beat), text=chord_text))
      
      # de-duplication
      prev_chord = None
      dedup_chords = []
      for m in markers:
        if m.text != prev_chord:
          prev_chord = m.text
          dedup_chords.append(m)

      # return midi
      midi_obj.markers = dedup_chords
    return midi_obj

  def _pruning_grouped_notes_from_quantization(self, instr_grid:dict):
    '''
    In case where notes are grouped in the same quant_time but with different start time, unintentional chords are created
    rule1: if notes have half step interval, delete the shorter one
    rule2: if notes do not share 50% of duration of the shorter note, delete the shorter one
    '''
    for instr in instr_grid.keys():
      time_list = sorted(list(instr_grid[instr].keys()))
      for time in time_list:
        notes = instr_grid[instr][time]
        if len(notes) == 1:
          continue
        else:
          new_notes = []
        # sort in pitch with ascending order
        notes.sort(key=lambda x: x.pitch)
        for i in range(len(notes)-1):
          # if start time is same add to new_notes
          if notes[i].start == notes[i+1].start:
            new_notes.append(notes[i])
            new_notes.append(notes[i+1])
            continue
          if notes[i].pitch == notes[i+1].pitch or notes[i].pitch + 1 == notes[i+1].pitch:
            # select longer note
            if notes[i].end - notes[i].start > notes[i+1].end - notes[i+1].start:
              new_notes.append(notes[i])
            else:
              new_notes.append(notes[i+1])
          else:
            # check how much duration they share
            shared_duration = min(notes[i].end, notes[i+1].end) - max(notes[i].start, notes[i+1].start)
            shorter_duration = min(notes[i].end - notes[i].start, notes[i+1].end - notes[i+1].start)
            # unless they share more than 80% of duration, select longer note (pruning shorter note)
            if shared_duration / shorter_duration < 0.8:
              if notes[i].end - notes[i].start > notes[i+1].end - notes[i+1].start:
                new_notes.append(notes[i])
              else:
                new_notes.append(notes[i+1])
            else:
              if len(new_notes) == 0:
                new_notes.append(notes[i])
                new_notes.append(notes[i+1])
              else:
                new_notes.append(notes[i+1])
        instr_grid[instr][time] = new_notes

  def _midi2corpus(self, midi_obj:miditoolkit.midi.parser.MidiFile):
    # Checks if the ticks per beat in the MIDI file is lower than the expected resolution.
    # If it is, raise an error.
    if midi_obj.ticks_per_beat < self.in_beat_resolution:
      raise ValueError(f'[x] Irregular ticks_per_beat. {midi_obj.ticks_per_beat}')

    # Ensure there is at least one time signature change in the MIDI file.
    if len(midi_obj.time_signature_changes) == 0:
      raise ValueError('[x] No time_signature_changes')
    
    # Ensure there are no duplicated time signature changes.
    time_list = [ts.time for ts in midi_obj.time_signature_changes]
    if len(time_list) != len(set(time_list)):
      raise ValueError('[x] Duplicated time_signature_changes')
    
    # If the dataset is 'LakhClean' or 'SymphonyMIDI', verify there are at least 4 tracks.
    if self.dataset_name == 'LakhClean' or self.dataset_name == 'SymphonyMIDI':
      if len(midi_obj.instruments) < 4:
        raise ValueError('[x] We will use more than 4 tracks in Lakh Clean dataset.')
    
    # Calculate the resolution of ticks per beat as a fraction.
    in_beat_tick_resol = Fraction(midi_obj.ticks_per_beat, self.in_beat_resolution)
    
    # Extract the initial time signature (numerator and denominator) and calculate the number of ticks for the first bar.
    initial_numerator = midi_obj.time_signature_changes[0].numerator
    initial_denominator = midi_obj.time_signature_changes[0].denominator
    first_bar_resol = int(midi_obj.ticks_per_beat * initial_numerator * (4 / initial_denominator))

    # --- load notes --- #
    instr_notes = self._make_instr_notes(midi_obj)
    
    # --- load information --- #
    # load chords, labels
    chords = split_markers(midi_obj.markers)
    chords.sort(key=lambda x: x.time)

    # load tempos
    tempos = midi_obj.tempo_changes
    tempos.sort(key=lambda x: x.time)

    # --- process items to grid --- #
    # compute empty bar offset at head
    first_note_time = min([instr_notes[k][0].start for k in instr_notes.keys()])
    last_note_time = max([instr_notes[k][-1].start for k in instr_notes.keys()])

    quant_time_first = int(round(first_note_time  / in_beat_tick_resol)) * in_beat_tick_resol
    offset = quant_time_first // first_bar_resol # empty bar
    offset_by_resol = offset * first_bar_resol

    # --- process notes --- #
    instr_grid = dict()
    for key in instr_notes.keys():
      notes = instr_notes[key]
      note_grid = defaultdict(list)
      for note in notes:
        # skip notes out of range, below C-1 and above C8
        if note.pitch < 12 or note.pitch >= 120:
          continue

        # in case when the first note starts at slightly before the first bar
        note.start = note.start - offset_by_resol if note.start - offset_by_resol > 0 else 0
        note.end = note.end - offset_by_resol if note.end - offset_by_resol > 0 else 0

        # relative duration
        # skip note with 0 duration
        note_duration = note.end - note.start
        relative_duration = round(note_duration / in_beat_tick_resol)
        if relative_duration == 0:
          continue
        if relative_duration > self.in_beat_resolution * 8: # 8 beats
          relative_duration = self.in_beat_resolution * 8
          
        # use regular duration bins
        note.quantized_duration = self.regular_duration_bins[np.argmin(abs(self.regular_duration_bins-relative_duration))]

        # quantize start time 
        quant_time = int(round(note.start / in_beat_tick_resol)) * in_beat_tick_resol

        # velocity
        note.velocity = self.regular_velocity_bins[
          np.argmin(abs(self.regular_velocity_bins-note.velocity))]

        # append
        note_grid[quant_time].append(note)

      # set to track
      instr_grid[key] = note_grid
    
    # --- pruning grouped notes --- #
    self._pruning_grouped_notes_from_quantization(instr_grid)
  
    # --- process chords --- #
    chord_grid = defaultdict(list)
    for chord in chords:
      # quantize
      chord.time = chord.time - offset_by_resol
      chord.time  = 0 if chord.time < 0 else chord.time
      quant_time = int(round(chord.time / in_beat_tick_resol)) * in_beat_tick_resol
      chord_grid[quant_time].append(chord)

    # --- process tempos --- #
    first_notes_list = []
    for instr in instr_grid.keys():
      time_list = sorted(list(instr_grid[instr].keys()))
      first_quant_time = time_list[0]
      first_notes_list.append(first_quant_time)
    quant_first_note_time = min(first_notes_list)

    tempo_grid = defaultdict(list)
    for tempo in tempos:
      # quantize
      tempo.time = tempo.time - offset_by_resol if tempo.time - offset_by_resol > 0 else 0
      quant_time = int(round(tempo.time / in_beat_tick_resol)) * in_beat_tick_resol
      tempo.tempo = self.regular_tempo_bins[
        np.argmin(abs(self.regular_tempo_bins-tempo.tempo))]
      if quant_time < quant_first_note_time:
        tempo_grid[quant_first_note_time].append(tempo)
      else:
        tempo_grid[quant_time].append(tempo)
    if len(tempo_grid[quant_first_note_time]) > 1:
      tempo_grid[quant_first_note_time] = [tempo_grid[quant_first_note_time][-1]]
    
    # --- process time signature --- #
    quant_time_signature = deepcopy(midi_obj.time_signature_changes)
    quant_time_signature.sort(key=lambda x: x.time)
    for ts in quant_time_signature:
      ts.time = ts.time - offset_by_resol if ts.time - offset_by_resol > 0 else 0
      ts.time = int(round(ts.time / in_beat_tick_resol)) * in_beat_tick_resol

    # --- make new midi object to check processed values --- #
    new_midi_obj = miditoolkit.midi.parser.MidiFile()
    new_midi_obj.ticks_per_beat = midi_obj.ticks_per_beat
    new_midi_obj.max_tick = midi_obj.max_tick
    for instr_idx in instr_grid.keys():
      new_instrument = Instrument(program=instr_idx)
      new_instrument.notes = [y for x in instr_grid[instr_idx].values() for y in x]
      new_midi_obj.instruments.append(new_instrument)
    new_midi_obj.markers = [y for x in chord_grid.values() for y in x]
    new_midi_obj.tempo_changes = [y for x in tempo_grid.values() for y in x]
    new_midi_obj.time_signature_changes = midi_obj.time_signature_changes

    # make corpus
    song_data = {
      'notes': instr_grid,
      'chords': chord_grid,
      'tempos': tempo_grid,
      'metadata': {
        'first_note': first_note_time,
        'last_note': last_note_time,
        'time_signature': quant_time_signature,
        'ticks_per_beat': midi_obj.ticks_per_beat,
        }
      }
    return song_data, new_midi_obj

  def _make_instr_notes(self, midi_obj):
    '''
    This part is important, we can use three different ways to merge instruments
    1st option: compare the number of notes and choose tracks with more notes
    2nd option: merge all instruments with the same tracks
    3rd option: leave all instruments as they are. differentiate tracks with different track number
    
    In this version we choose to use the 2nd option as it helps to reduce the number of tracks and sequence length
    '''
    instr_notes = defaultdict(list)
    for instr in midi_obj.instruments:
      instr_idx = instr.program
      # change instrument idx
      instr_name = PROGRAM_INSTRUMENT_MAP.get(instr_idx)
      if instr_name is None:
        continue
      new_instr_idx = INSTRUMENT_PROGRAM_MAP[instr_name]
      instr_notes[new_instr_idx].extend(instr.notes)
      instr_notes[new_instr_idx].sort(key=lambda x: (x.start, -x.pitch))
    return instr_notes

  # refered to SymphonyNet "https://github.com/symphonynet/SymphonyNet"
  def _merge_percussion(self, midi_obj:miditoolkit.midi.parser.MidiFile):
    '''
    merge drum track to one track
    '''
    drum_0_lst = []
    new_instruments = []
    for instrument in midi_obj.instruments:
      if len(instrument.notes) == 0:
        continue
      if instrument.is_drum:
        drum_0_lst.extend(instrument.notes)
      else:
        new_instruments.append(instrument)
    if len(drum_0_lst) > 0:
      drum_0_lst.sort(key=lambda x: x.start)
      # remove duplicate
      drum_0_lst = list(k for k, _ in itertools.groupby(drum_0_lst))
      drum_0_instrument = Instrument(program=114, is_drum=True, name="percussion")
      drum_0_instrument.notes = drum_0_lst
      new_instruments.append(drum_0_instrument)
    midi_obj.instruments = new_instruments

  # referred to mmt "https://github.com/salu133445/mmt"
  def _pruning_instrument(self, midi_obj:miditoolkit.midi.parser.MidiFile):
    '''
    merge instrument number with similar intrument category
    ex. 0: Acoustic Grand Piano, 1: Bright Acoustic Piano, 2: Electric Grand Piano into 0: Acoustic Grand Piano
    '''
    new_instruments = []
    for instr in midi_obj.instruments:
      instr_idx = instr.program
      # change instrument idx
      instr_name = PROGRAM_INSTRUMENT_MAP.get(instr_idx)
      if instr_name != None:
        new_instruments.append(instr)
    midi_obj.instruments = new_instruments

  # refered to SymphonyNet "https://github.com/symphonynet/SymphonyNet"
  def _limit_max_track(self, midi_obj:miditoolkit.midi.parser.MidiFile, MAX_TRACK:int=16):
      '''
      merge track with least notes to other track with same program
      and limit the maximum amount of track to 16
      '''
      if len(midi_obj.instruments) == 1:
        if midi_obj.instruments[0].is_drum:
          midi_obj.instruments[0].program = 114
          midi_obj.instruments[0].is_drum = False
        return midi_obj
      good_instruments = midi_obj.instruments
      good_instruments.sort(
          key=lambda x: (not x.is_drum, -len(x.notes)))  # place drum track or the most note track at first
      assert good_instruments[0].is_drum == True or len(good_instruments[0].notes) >= len(
          good_instruments[1].notes), tuple(len(x.notes) for x in good_instruments[:3])
      # assert good_instruments[0].is_drum == False, (, len(good_instruments[2]))
      track_idx_lst = list(range(len(good_instruments)))
      if len(good_instruments) > MAX_TRACK:
          new_good_instruments = copy.deepcopy(good_instruments[:MAX_TRACK])
          # print(midi_file_path)
          for id in track_idx_lst[MAX_TRACK:]:
              cur_ins = good_instruments[id]
              merged = False
              new_good_instruments.sort(key=lambda x: len(x.notes))
              for nid, ins in enumerate(new_good_instruments):
                  if cur_ins.program == ins.program and cur_ins.is_drum == ins.is_drum:
                      new_good_instruments[nid].notes.extend(cur_ins.notes)
                      merged = True
                      break
              if not merged:
                  pass
          good_instruments = new_good_instruments

      assert len(good_instruments) <= MAX_TRACK, len(good_instruments)
      for idx, good_instrument in enumerate(good_instruments):
          if good_instrument.is_drum:
              good_instruments[idx].program = 114
              good_instruments[idx].is_drum = False
      midi_obj.instruments = good_instruments

  def _pruning_notes_for_chord_extraction(self, midi_obj:miditoolkit.midi.parser.MidiFile):
    '''
    extract notes for chord extraction
    '''
    new_midi_obj = miditoolkit.midi.parser.MidiFile()
    new_midi_obj.ticks_per_beat = midi_obj.ticks_per_beat
    new_midi_obj.max_tick = midi_obj.max_tick
    new_instrument = Instrument(program=0, is_drum=False, name="for_chord")
    new_instruments = []
    new_notes = []
    for instrument in midi_obj.instruments:
      if instrument.program == 114 or instrument.is_drum: # pass drum track
        continue
      valid_notes = [note for note in instrument.notes if note.pitch >= 21 and note.pitch <= 108]
      new_notes.extend(valid_notes)
    new_notes.sort(key=lambda x: x.start)
    new_instrument.notes = new_notes
    new_instruments.append(new_instrument)
    new_midi_obj.instruments = new_instruments
    return new_midi_obj

def get_argument_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "-d",
      "--dataset",
      required=True,
      choices=("BachChorale", "Pop1k7", "Pop909", "SOD", "LakhClean", "SymphonyMIDI"),
      type=str,
      help="dataset names",
  )
  parser.add_argument(
      "-f",
      "--num_features",
      required=True,
      choices=(4, 5, 7, 8),
      type=int,
      help="number of features",
  )
  parser.add_argument(
      "-i",
      "--in_dir",
      default="../dataset/MIDI_dataset/",
      type=Path,
      help="input data directory",
  )
  parser.add_argument(
      "-o",
      "--out_dir",
      default="../dataset/represented_data/corpus/",
      type=Path,
      help="output data directory",
  )
  parser.add_argument(
      "--debug",
      action="store_true",
      help="enable debug mode",
  )
  return parser

def main():
  parser = get_argument_parser()
  args = parser.parse_args()
  corpus_maker = CorpusMaker(args.dataset, args.num_features, args.in_dir, args.out_dir, args.debug)
  corpus_maker.make_corpus()

if __name__ == "__main__":
  main()

