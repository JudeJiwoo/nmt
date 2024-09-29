from typing import Any
from fractions import Fraction
from collections import defaultdict

from miditoolkit import TimeSignature

from constants import *

def frange(start, stop, step):
  while start < stop:
    yield start
    start += step


################################# for REMI style encoding #################################

class Corpus2event_remi():
  def __init__(self, num_features):
    self.num_features = num_features
  
  def _create_event(self, name, value):
    event = dict()
    event['name'] = name
    event['value'] = value
    return event
  
  def _normalize_signature(self, time_signature, ticks_per_beat, next_change_point):
      if (time_signature.numerator, time_signature.denominator) in REGULAR_NUM_DENOM:
        return [time_signature]
      numerator, denominator, bar_start_tick = time_signature.numerator, time_signature.denominator, time_signature.time
      # 16th, 32th note based time signatures normalized to 6/4
      if denominator == 16 or denominator == 32 or denominator == 64:
        return [TimeSignature(4, 4, time_signature.time)]
      elif denominator == 6 and numerator == 4:
        return [TimeSignature(3, 4, time_signature.time), TimeSignature(3, 4, time_signature.time)]
      # determine which regular signatures can be used for the current denominator
      possible_signatures = [sig for sig in CORE_NUM_DENOM if sig[1] == denominator]
      # sort by numerator in descending order to use large numerators first
      possible_signatures.sort(key=lambda x: x[0], reverse=True)
      result = []
      while numerator > 0:
        for sig in possible_signatures:
          while numerator >= sig[0]:
            result.append(sig)
            numerator -= sig[0]
      # calculate bar resolution of each bar
      bar_resol_list = [int(ticks_per_beat * numerator * (4 / denominator)) for numerator, denominator in result]
      # select bars that fit in the current bar
      total_length = 0
      for idx, bar_resol in enumerate(bar_resol_list):
        total_length += bar_resol
        if total_length > next_change_point - bar_start_tick:
          result = result[:idx+1]
          break
      # in case total length is too short, repeat the first bar(larger)
      while total_length < next_change_point - bar_start_tick:
        result.append(result[0])
        total_length += int(ticks_per_beat * result[0][0] * (4 / result[0][1]))
      # change to TimeSignature object
      bar_resol_list = [int(ticks_per_beat * numerator * (4 / denominator)) for numerator, denominator in result]
      bar_resol_list.insert(0, 0)
      total_length = bar_start_tick
      normalized_result = []
      for sig, length in zip(result, bar_resol_list):
        total_length += length
        normalized_result.append(TimeSignature(sig[0], sig[1], total_length))
      return normalized_result

  def _process_time_signature(self, time_signature_changes, ticks_per_beat, first_note_tick, global_end):
    # check validity
    if len(time_signature_changes) == 0:
      print("No time signature change in this tune")
      return None
    if time_signature_changes[0].time != 0 and time_signature_changes[0].time > first_note_tick:
      print("The first time signature change is not at the beginning of the tune")
      return None
    # de-duplication
    processed_time_signature_changes = []
    for idx, time_sig in enumerate(time_signature_changes):
      if idx == 0:
        processed_time_signature_changes.append(time_sig)
      else:
        prev_time_sig = time_signature_changes[idx-1]
        if not (prev_time_sig.numerator == time_sig.numerator and prev_time_sig.denominator == time_sig.denominator):
          processed_time_signature_changes.append(time_sig)
    # normalize time signature
    normalized_time_signature_changes = []
    for idx, time_signature in enumerate(processed_time_signature_changes):
      if idx == len(time_signature_changes) - 1:
        next_change_point = global_end
      else:
        next_change_point = time_signature_changes[idx+1].time
      normalized_time_signature_changes.extend(self._normalize_signature(time_signature, ticks_per_beat, next_change_point))
    time_signature_changes = normalized_time_signature_changes
    return time_signature_changes

  def __call__(self, song_data, in_beat_resolution):
    '''
    This is corpus2event_remi_v2 provided by CP paper
    '''
    # --- global tag --- #
    first_note_tick = song_data['metadata']['first_note']
    global_end = song_data['metadata']['last_note']
    time_signature_changes = song_data['metadata']['time_signature']
    ticks_per_beat = song_data['metadata']['ticks_per_beat']
    in_beat_tick_resol = Fraction(ticks_per_beat, in_beat_resolution) # 1024/12 -> (256, 3)

    # --- process time signature --- #
    time_signature_changes = self._process_time_signature(time_signature_changes, ticks_per_beat, first_note_tick, global_end)
    if time_signature_changes == None:
      return None

    # --- create sequence --- #
    prev_instr_idx = None
    final_sequence = []
    final_sequence.append(self._create_event('SOS', None))
    prev_chord = None
    prev_tempo = None
    chord_value = None
    tempo_value = None

    for idx in range(len(time_signature_changes)):
      time_sig_change_flag = True
      # calculate bar resolution
      numerator = time_signature_changes[idx].numerator
      denominator = time_signature_changes[idx].denominator
      time_sig_name = f'time_signature_{numerator}/{denominator}'
      bar_resol = int(ticks_per_beat * numerator * (4 / denominator))
      bar_start_tick = time_signature_changes[idx].time
      if idx == len(time_signature_changes) - 1:
        next_change_point = global_end
      else:
        next_change_point = time_signature_changes[idx+1].time
      for measure_step in frange(bar_start_tick, next_change_point, bar_resol):
        empty_bar_token = self._create_event('Bar', None)
        # check consecutive empty bars (more than 4 is not allowed)
        if len(final_sequence) >= 4:
          if not (final_sequence[-1] == empty_bar_token and final_sequence[-2] == empty_bar_token and final_sequence[-3] == empty_bar_token and final_sequence[-4] == empty_bar_token):
            if time_sig_change_flag:
              final_sequence.append(self._create_event('Bar', time_sig_name))
            else:
              final_sequence.append(self._create_event('Bar', None))
          else:
            if time_sig_change_flag:
              final_sequence.append(self._create_event('Bar', time_sig_name))
        else:
          if time_sig_change_flag:
            final_sequence.append(self._create_event('Bar', time_sig_name))
          else:
            final_sequence.append(self._create_event('Bar', None))
        time_sig_change_flag = False
        for in_beat_off_idx, beat_step in enumerate(frange(measure_step, measure_step + bar_resol, in_beat_tick_resol)):
          events_list = []
          # unpack
          t_chords = song_data['chords'].get(beat_step)
          t_tempos = song_data['tempos'].get(beat_step)

          # chord & tempo
          if self.num_features == 8 or self.num_features == 7:
            if t_chords != None:
              root, quality, _ = t_chords[-1].text.split('_')
              chord_value = root + '_' + quality
            if t_tempos != None:
              tempo_value = t_tempos[-1].tempo

          # we need to consider the duplicated notes
          instrument_note_dict = defaultdict(set)
          for instrument_idx in song_data['notes'].keys():
            t_notes = song_data['notes'][instrument_idx].get(beat_step)
            # note
            if t_notes != None:
              # add chord & tempo
              if prev_chord != chord_value:
                events_list.append(self._create_event('Chord', chord_value))
                prev_chord = chord_value
              if prev_tempo != tempo_value:
                events_list.append(self._create_event('Tempo', tempo_value))
                prev_tempo = tempo_value

              # add instrument_idx
              if self.num_features == 8 or self.num_features == 5:
                if prev_instr_idx != instrument_idx:
                  events_list.append(self._create_event('Instrument', instrument_idx))
                  prev_instr_idx = instrument_idx
              for note in t_notes:
                if (note.pitch, note.quantized_duration) not in instrument_note_dict[instrument_idx]:
                  instrument_note_dict[instrument_idx].add((note.pitch, note.quantized_duration))
                  events_list.extend([
                    self._create_event('Note_Pitch', note.pitch),
                    self._create_event('Note_Duration', note.quantized_duration),
                  ])
                  if self.num_features == 8 or self.num_features == 7:
                    events_list.append(self._create_event('Note_Velocity', note.velocity))
          # collect & beat
          if len(events_list):
            final_sequence.append(self._create_event('Beat', in_beat_off_idx))
            final_sequence.extend(events_list)
    # --- end with BAR & EOS --- #
    final_sequence.append(self._create_event('Bar', None))
    final_sequence.append(self._create_event('EOS', None))
    return final_sequence

################################# for CP style encoding #################################
  
class Corpus2event_cp(Corpus2event_remi):
  def __init__(self, num_features):
    super().__init__(num_features)
    self.num_features = num_features
    self._init_event_template()
    
  def _init_event_template(self):
    '''
    The order of musical features is changed into: N, B, C, T, I, P, D, V, same as the order of REMI.
    Unlike the original CP, we need to consider sequential prediction of musical features
    '''
    self.event_template = {}
    if self.num_features == 8:
      feature_names = ['type', 'beat', 'chord', 'tempo', 'instrument', 'pitch', 'duration', 'velocity']
    elif self.num_features == 7:
      feature_names = ['type', 'beat', 'chord', 'tempo', 'pitch', 'duration', 'velocity']
    elif self.num_features == 5:
      feature_names = ['type', 'beat', 'instrument', 'pitch', 'duration']
    elif self.num_features == 4:
      feature_names = ['type', 'beat', 'pitch', 'duration']
    for feature_name in feature_names:
      self.event_template[feature_name] = 0

  def create_cp_metrical_event(self, pos, chord, tempo):
    meter_event = self.event_template.copy()
    meter_event['type'] = 'Metrical'
    meter_event['beat'] = pos
    if self.num_features == 7 or self.num_features == 8:
      meter_event['chord'] = chord
      meter_event['tempo'] = tempo
    return meter_event

  def create_cp_note_event(self, instrument_name, pitch, duration, velocity):
    note_event = self.event_template.copy()
    note_event['type'] = 'Note'
    note_event['pitch'] = pitch
    note_event['duration'] = duration
    if self.num_features == 5 or self.num_features == 8:
      note_event['instrument'] = instrument_name
    if self.num_features == 7 or self.num_features == 8:
      note_event['velocity'] = velocity
    return note_event

  def create_cp_bar_event(self, time_sig_change_flag=False, time_sig_name=None):
    meter_event = self.event_template.copy()
    if time_sig_change_flag:
      meter_event['type'] = time_sig_name
      meter_event['beat'] = 'Bar'
    else:
      meter_event['type'] = 'Metrical'
      meter_event['beat'] = 'Bar'
    return meter_event

  def __call__(self, song_data, in_beat_resolution):
    '''
    intput is loaded data from pickle file
    we also added N_N for the first note of the tune when there is no previous tempo & chord info
    the original CP deals with this case by using CONTI
    '''
    # --- global tag --- #
    first_note_tick = song_data['metadata']['first_note']
    global_end = song_data['metadata']['last_note']
    time_signature_changes = song_data['metadata']['time_signature']
    ticks_per_beat = song_data['metadata']['ticks_per_beat']
    in_beat_tick_resol = Fraction(ticks_per_beat, in_beat_resolution) # 1024/12 -> (256, 3)

    # --- process time signature --- #
    time_signature_changes = self._process_time_signature(time_signature_changes, ticks_per_beat, first_note_tick, global_end)
    if time_signature_changes == None:
      return None
  
    # --- create sequence --- #
    final_sequence = []
    final_sequence.append(self.create_cp_sos_event())
    chord_text = None
    tempo_text = None
    for idx in range(len(time_signature_changes)):
      time_sig_change_flag = True
      # calculate bar resolution
      numerator = time_signature_changes[idx].numerator
      denominator = time_signature_changes[idx].denominator
      time_sig_name = f'time_signature_{numerator}/{denominator}'
      bar_resol = int(ticks_per_beat * numerator * (4 / denominator))
      bar_start_tick = time_signature_changes[idx].time
      if idx == len(time_signature_changes) - 1:
        next_change_point = global_end
      else:
        next_change_point = time_signature_changes[idx+1].time
      for measure_step in frange(bar_start_tick, next_change_point, bar_resol):
        empty_bar_token = self.create_cp_bar_event()
        # check consecutive empty bars (more than 4 is not allowed)
        if len(final_sequence) >= 4:
          if not (final_sequence[-1] == empty_bar_token and final_sequence[-2] == empty_bar_token and final_sequence[-3] == empty_bar_token and final_sequence[-4] == empty_bar_token):
            final_sequence.append(self.create_cp_bar_event(time_sig_change_flag, time_sig_name))
          else:
            if time_sig_change_flag:
              final_sequence.append(self.create_cp_bar_event(time_sig_change_flag, time_sig_name))
        else:
          final_sequence.append(self.create_cp_bar_event(time_sig_change_flag, time_sig_name))
        time_sig_change_flag = False
        for in_beat_off_idx, beat_step in enumerate(frange(measure_step, measure_step + bar_resol, in_beat_tick_resol)):
          chord_tempo_flag = False
          events_list = []
          pos_text = 'Beat_' + str(in_beat_off_idx)

          # chord & tempo
          # unpack
          t_chords = song_data['chords'].get(beat_step)
          t_tempos = song_data['tempos'].get(beat_step)
          # chord
          if t_chords != None:
            root, quality, bass = t_chords[-1].text.split('_')
            chord_text = 'Chord_' + root + '_' + quality
          # tempo
          if t_tempos != None:
            tempo_text = 'Tempo_' + str(t_tempos[-1].tempo)
          
          # instrument & note
          for instrument_idx in song_data['notes'].keys():
            instrument_name = f"Instrument_{int(instrument_idx)}"
            t_notes = song_data['notes'][instrument_idx].get(beat_step)
                  
            # note
            if t_notes != None:
              if not chord_tempo_flag:
                if chord_text == None:
                  chord_text = 'Chord_N_N'
                if tempo_text == None:
                  tempo_text = 'Tempo_N_N'
                events_list.append(self.create_cp_metrical_event(pos_text, chord_text, tempo_text))
                chord_tempo_flag = True
              for note in t_notes:
                note_pitch_text = 'Note_Pitch_' + str(note.pitch)
                note_duration_text = 'Note_Duration_' + str(note.quantized_duration)
                note_velocity_text = 'Note_Velocity_' + str(note.velocity)
                events_list.append(self.create_cp_note_event(instrument_name, note_pitch_text, note_duration_text, note_velocity_text))

          # collect & beat
          if len(events_list) > 0:
            final_sequence.extend(events_list)
    # --- end with BAR & EOS --- #
    final_sequence.append(self.create_cp_bar_event())
    final_sequence.append(self.create_cp_eos_event())
    return final_sequence
  
################################# for NB style encoding #################################

class Corpus2event_nb(Corpus2event_cp):
  def __init__(self, num_features):
    super().__init__(num_features)
    self.num_features = num_features
    self._init_event_template()
    
  def _init_event_template(self):
    self.event_template = {}
    if self.num_features == 8:
      feature_names = ['type', 'beat', 'chord', 'tempo', 'instrument', 'pitch', 'duration', 'velocity']
    elif self.num_features == 7:
      feature_names = ['type', 'beat', 'chord', 'tempo', 'pitch', 'duration', 'velocity']
    elif self.num_features == 5:
      feature_names = ['type', 'beat', 'instrument', 'pitch', 'duration']
    elif self.num_features == 4:
      feature_names = ['type', 'beat', 'pitch', 'duration']
    for feature_name in feature_names:
      self.event_template[feature_name] = 0

  def create_nb_event(self, bar_beat_type, pos, chord, tempo, instrument_name, pitch, duration, velocity):
    total_event = self.event_template.copy()
    total_event['type'] = bar_beat_type
    total_event['beat'] = pos
    total_event['pitch'] = pitch
    total_event['duration'] = duration
    if self.num_features == 5 or self.num_features == 8:
      total_event['instrument'] = instrument_name
    if self.num_features == 7 or self.num_features == 8:
      total_event['chord'] = chord
      total_event['tempo'] = tempo
      total_event['velocity'] = velocity
    return total_event

  def create_nb_empty_bar_event(self):
    total_event = self.event_template.copy()
    total_event['type'] = 'Empty_Bar'
    return total_event

  def get_bar_beat_idx(self, bar_flag, beat_flag, time_sig_name, time_sig_change_flag):
    if time_sig_change_flag: # new time signature
      return "NNN_" + time_sig_name
    else:
      if bar_flag and beat_flag: # same time sig & new bar & new beat
        return "SNN"
      elif not bar_flag and beat_flag: # same time sig & same bar & new beat
        return "SSN"
      elif not bar_flag and not beat_flag: # same time sig & same bar & same beat
        return "SSS"

  def __call__(self, song_data, in_beat_resolution:int):
    '''
    intput is loaded data from pickle file
    '''
    # --- global tag --- #
    first_note_tick = song_data['metadata']['first_note']
    global_end = song_data['metadata']['last_note']
    time_signature_changes = song_data['metadata']['time_signature']
    ticks_per_beat = song_data['metadata']['ticks_per_beat']
    in_beat_tick_resol = Fraction(ticks_per_beat, in_beat_resolution) # 1024/12 -> (256, 3)

    # --- process time signature --- #
    time_signature_changes = self._process_time_signature(time_signature_changes, ticks_per_beat, first_note_tick, global_end)
    if time_signature_changes == None:
      return None
      
    # --- create sequence --- #
    final_sequence = []
    final_sequence.append(self.create_nb_sos_event())
    chord_text = None
    tempo_text = None
    for idx in range(len(time_signature_changes)):
      time_sig_change_flag = True
      # calculate bar resolution
      numerator = time_signature_changes[idx].numerator
      denominator = time_signature_changes[idx].denominator
      time_sig_name = f'time_signature_{numerator}/{denominator}'
      bar_resol = int(ticks_per_beat * numerator * (4 / denominator))
      bar_start_tick = time_signature_changes[idx].time
      if idx == len(time_signature_changes) - 1:
        next_change_point = global_end
      else:
        next_change_point = time_signature_changes[idx+1].time
      for measure_step in frange(bar_start_tick, next_change_point, bar_resol):
        bar_flag = True
        note_flag = False
        for in_beat_off_idx, beat_step in enumerate(frange(measure_step, measure_step + bar_resol, in_beat_tick_resol)):
          beat_flag = True
          events_list = []
          pos_text = 'Beat_' + str(in_beat_off_idx)

          # chord & tempo
          # unpack
          t_chords = song_data['chords'].get(beat_step)
          t_tempos = song_data['tempos'].get(beat_step)
          # chord
          if t_chords != None:
            root, quality, bass = t_chords[-1].text.split('_')
            chord_text = 'Chord_' + root + '_' + quality
          # tempo
          if t_tempos != None:
            tempo_text = 'Tempo_' + str(t_tempos[-1].tempo)
          
          # instrument & note
          for instrument_idx in song_data['notes'].keys():
            instrument_name = f"Instrument_{int(instrument_idx)}"
            t_notes = song_data['notes'][instrument_idx].get(beat_step)

            # note
            if t_notes != None:
              note_flag = True
              if chord_text == None:
                chord_text = 'Chord_N_N'
              if tempo_text == None:
                tempo_text = 'Tempo_N_N'
              for note in t_notes:
                note_pitch_text = 'Note_Pitch_' + str(note.pitch)
                note_duration_text = 'Note_Duration_' + str(note.quantized_duration)
                note_velocity_text = 'Note_Velocity_' + str(note.velocity)
                bar_beat_type = self.get_bar_beat_idx(bar_flag, beat_flag, time_sig_name, time_sig_change_flag)
                events_list.append(self.create_nb_event(bar_beat_type, pos_text, chord_text, tempo_text, instrument_name, note_pitch_text, note_duration_text, note_velocity_text))
                bar_flag = False
                beat_flag = False
                time_sig_change_flag = False

          # collect & beat
          if events_list != None and len(events_list):
            final_sequence.extend(events_list)
        # when there is no note in this bar
        if not note_flag:
          # avoid consecutive empty bars (more than 4 is not allowed)
          empty_bar_token = self.create_nb_empty_bar_event()
          if len(final_sequence) >= 4:
            if final_sequence[-1] == empty_bar_token and final_sequence[-2] == empty_bar_token and final_sequence[-3] == empty_bar_token and final_sequence[-4] == empty_bar_token:
              continue
          final_sequence.append(empty_bar_token)
    # EOS
    final_sequence.append(self.create_nb_eos_event())
    return final_sequence