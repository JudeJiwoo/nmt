from typing import Any
from fractions import Fraction
from collections import defaultdict

from miditoolkit import TimeSignature

from constants import *

'''
This script contains specific encoding functions for different encoding schemes.
'''

def frange(start, stop, step):
  while start < stop:
    yield start
    start += step

################################# for REMI style encoding #################################

class Corpus2event_remi():
    def __init__(self, num_features:int):
      self.num_features = num_features
    
    def _create_event(self, name, value):
      event = dict()
      event['name'] = name
      event['value'] = value
      return event
  
    def _normalize_time_signature(self, time_signature, ticks_per_beat, next_change_point):
        """
        Normalize irregular time signatures to standard ones by breaking them down 
        into common time signatures, and adjusting their durations to fit the given 
        musical structure.

        Parameters:
        - time_signature: TimeSignature object with numerator, denominator, and start time.
        - ticks_per_beat: Number of ticks per beat, representing the resolution of the timing.
        - next_change_point: Tick position where the next time signature change occurs.

        Returns:
        - A list of TimeSignature objects, normalized to fit within regular time signatures.

        Procedure:
        1. If the time signature is already a standard one (in REGULAR_NUM_DENOM), return it.
        2. For non-standard signatures, break them down into simpler, well-known signatures.
          - For unusual denominations (e.g., 16th, 32nd, or 64th notes), normalize to 4/4.
          - For 6/4 signatures, break it into two 3/4 measures.
        3. If the time signature has a non-standard numerator and denominator, break it down 
          into the largest possible numerators that still fit within the denominator. 
          This ensures that the final measure fits within the regular time signature format.
        4. Calculate the resolution (duration in ticks) for each bar and ensure the bars 
          fit within the time until the next change point.
          - Adjust the number of bars if they exceed the available space.
          - If the total length is too short, repeat the first (largest) bar to fill the gap.
        5. Convert the breakdown into TimeSignature objects and return the normalized result.
        """
        
        # Check if the time signature is a regular one, return it if so
        if (time_signature.numerator, time_signature.denominator) in REGULAR_NUM_DENOM:
            return [time_signature]
        
        # Extract time signature components
        numerator, denominator, bar_start_tick = time_signature.numerator, time_signature.denominator, time_signature.time

        # Normalize time signatures with 16th, 32nd, or 64th note denominators to 4/4
        if denominator in [16, 32, 64]:
            return [TimeSignature(4, 4, time_signature.time)]
        
        # Special case for 6/4, break it into two 3/4 bars
        elif denominator == 6 and numerator == 4:
            return [TimeSignature(3, 4, time_signature.time), TimeSignature(3, 4, time_signature.time)]
        
        # Determine possible regular signatures for the given denominator
        possible_time_signatures = [sig for sig in CORE_NUM_DENOM if sig[1] == denominator]
        
        # Sort by numerator in descending order to prioritize larger numerators
        possible_time_signatures.sort(key=lambda x: x[0], reverse=True)
        
        result = []
        
        # Break down the numerator into smaller regular numerators
        while numerator > 0:
            for sig in possible_time_signatures:
                # Subtract numerators and add to the result
                while numerator >= sig[0]:
                    result.append(sig)
                    numerator -= sig[0]
        
        # Calculate the resolution (length in ticks) of each bar
        bar_resol_list = [int(ticks_per_beat * numerator * (4 / denominator)) for numerator, denominator in result]
        
        # Adjust bars to fit within the remaining ticks before the next change point
        total_length = 0
        for idx, bar_resol in enumerate(bar_resol_list):
            total_length += bar_resol
            if total_length > next_change_point - bar_start_tick:
                result = result[:idx+1]
                break
        
        # If the total length is too short, repeat the first (largest) bar until the gap is filled
        while total_length < next_change_point - bar_start_tick:
            result.append(result[0])
            total_length += int(ticks_per_beat * result[0][0] * (4 / result[0][1]))
        
        # Recalculate bar resolutions for the final result
        bar_resol_list = [int(ticks_per_beat * numerator * (4 / denominator)) for numerator, denominator in result]
        
        # Insert a starting resolution of 0 and calculate absolute tick positions for each TimeSignature
        bar_resol_list.insert(0, 0)
        total_length = bar_start_tick
        normalized_result = []
        for sig, length in zip(result, bar_resol_list):
            total_length += length
            normalized_result.append(TimeSignature(sig[0], sig[1], total_length))
        
        return normalized_result

    def _process_time_signature(self, time_signature_changes, ticks_per_beat, first_note_tick, global_end):
        """
        Process and normalize time signature changes for a given musical piece.

        Parameters:
        - time_signature_changes: A list of TimeSignature objects representing time signature changes in the music.
        - ticks_per_beat: The resolution of timing in ticks per beat.
        - first_note_tick: The tick position of the first note in the piece.
        - global_end: The tick position where the piece ends.

        Returns:
        - A list of processed and normalized time signature changes. If no valid time signature 
          changes are found, returns None.

        Procedure:
        1. Check the validity of the time signature changes:
          - Ensure there is at least one time signature change.
          - Ensure the first time signature change occurs at the beginning (before the first note).
        2. Remove duplicate consecutive time signatures:
          - Only add time signatures that differ from the previous one (de-duplication).
        3. Normalize the time signatures:
          - For each time signature, determine its duration by calculating the time until the 
            next change point or the end of the piece.
          - Use the _normalize_time_signature method to break down non-standard signatures into 
            simpler, well-known signatures that fit within the musical structure.
        4. Return the processed and normalized time signature changes.

        """
        
        # Check if there are any time signature changes
        if len(time_signature_changes) == 0:
            print("No time signature change in this tune")
            return None
        
        # Ensure the first time signature change is at the start of the piece (before the first note)
        if time_signature_changes[0].time != 0 and time_signature_changes[0].time > first_note_tick:
            print("The first time signature change is not at the beginning of the tune")
            return None
        
        # Remove consecutive duplicate time signatures (de-duplication)
        processed_time_signature_changes = []
        for idx, time_sig in enumerate(time_signature_changes):
            if idx == 0:
                processed_time_signature_changes.append(time_sig)
            else:
                prev_time_sig = time_signature_changes[idx-1]
                # Only add time signature if it's different from the previous one
                if not (prev_time_sig.numerator == time_sig.numerator and prev_time_sig.denominator == time_sig.denominator):
                    processed_time_signature_changes.append(time_sig)
        
        # Normalize the time signatures to standard formats
        normalized_time_signature_changes = []
        for idx, time_signature in enumerate(processed_time_signature_changes):
            if idx == len(time_signature_changes) - 1:
                # If it's the last time signature change, set the next change point as the end of the piece
                next_change_point = global_end
            else:
                # Otherwise, set the next change point as the next time signature's start time
                next_change_point = time_signature_changes[idx+1].time
            
            # Normalize the current time signature and extend the result
            normalized_time_signature_changes.extend(self._normalize_time_signature(time_signature, ticks_per_beat, next_change_point))
        
        # Return the list of processed and normalized time signatures
        time_signature_changes = normalized_time_signature_changes
        return time_signature_changes

    def _half_step_interval_gap_check_across_instruments(self, instrument_note_dict):
        '''
        This function checks for half-step interval gaps between notes across different instruments.
        It will avoid half-step intervals by keeping one note from any pair of notes that are a half-step apart, 
        regardless of which instrument they belong to.
        '''
        # order instrument_note_dict by pitch in descending order
        instrument_note_dict = dict(sorted(instrument_note_dict.items()))

        # Create a dictionary to store all pitches across instruments
        all_pitches = {}

        # Collect all pitches from each instrument and sort them in descending order
        for instrument, notes in instrument_note_dict.items():
            for pitch, durations in notes.items():
                all_pitches[pitch] = all_pitches.get(pitch, []) + [(instrument, durations)]

        # Sort the pitches in descending order
        sorted_pitches = sorted(all_pitches.keys(), reverse=True)

        # Create a new list to store the final pitches after comparison
        final_pitch_list = []

        # Use an index pointer to control the sliding window
        idx = 0
        while idx < len(sorted_pitches) - 1:
            current_pitch = sorted_pitches[idx]
            next_pitch = sorted_pitches[idx + 1]

            if current_pitch - next_pitch == 1:  # Check for a half-step interval gap
                current_max_duration = max(duration for _, durations in all_pitches[current_pitch] for duration, _ in durations)
                next_max_duration = max(duration for _, durations in all_pitches[next_pitch] for duration, _ in durations)

                if current_max_duration < next_max_duration:
                    # Keep the higher pitch (next_pitch) and skip the current_pitch
                    final_pitch_list.append(next_pitch)
                else:
                    # Keep the lower pitch (current_pitch) and skip the next_pitch
                    final_pitch_list.append(current_pitch)

                # Skip the next pitch because we already handled it
                idx += 2
            else:
                # No half-step gap, keep the current pitch and move to the next one
                final_pitch_list.append(current_pitch)
                idx += 1

        # Ensure the last pitch is added if it's not part of a half-step interval
        if idx == len(sorted_pitches) - 1:
            final_pitch_list.append(sorted_pitches[-1])

        # Filter out notes not in the final pitch list and update the instrument_note_dict
        for instrument in instrument_note_dict.keys():
            instrument_note_dict[instrument] = {
                pitch: instrument_note_dict[instrument][pitch]
                for pitch in sorted(instrument_note_dict[instrument].keys(), reverse=True) if pitch in final_pitch_list
            }

        return instrument_note_dict

    def __call__(self, song_data, in_beat_resolution):
        '''
        Process a song's data to generate a sequence of musical events, including bars, chords, tempo, 
        and notes, similar to the approach used in the CP paper (corpus2event_remi_v2).

        Parameters:
        - song_data: A dictionary containing metadata, notes, chords, and tempos of the song.
        - in_beat_resolution: The resolution of timing in beats (how many divisions per beat).

        Returns:
        - A sequence of musical events including start (SOS), bars, chords, tempo, instruments, notes,
          and an end (EOS) event. If the time signature is invalid, returns None.

        Procedure:
        1. **Global Setup**:
          - Extract global metadata like first and last note ticks, time signature changes, and ticks 
            per beat.
          - Compute `in_beat_tick_resol`, the ratio of ticks per beat to the input beat resolution, 
            to assist in dividing bars later.
          - Get a sorted list of instruments in the song.

        2. **Time Signature Processing**:
          - Call `_process_time_signature` to clean up and normalize the time signatures in the song.
          - If the time signatures are invalid (e.g., no time signature changes or missing at the 
            start), the function exits early with None.

        3. **Sequence Generation**:
          - Initialize the sequence with a start token (SOS) and prepare variables for tracking 
            previous chord, tempo, and instrument states.
          - Loop through each time signature change, dividing the song into measures based on the 
            current time signature's numerator and denominator.
          - For each measure, append "Bar" tokens to mark measure boundaries, while ensuring that no 
            more than four consecutive empty bars are added.
          - For each step within a measure, process the following:
            - **Chords**: If there is a chord change, add a corresponding chord event.
            - **Tempo**: If the tempo changes, add a tempo event.
            - **Notes**: Iterate over each instrument, adding notes and checking for half-step 
              intervals, deduplicating notes, and choosing the longest duration for each pitch.
          - Append a "Beat" event for each step with musical events.

        4. **End Sequence**:
          - Conclude the sequence by appending a final "Bar" token followed by an end token (EOS).
        '''

        # --- global tag --- #
        first_note_tick = song_data['metadata']['first_note']  # Starting tick of the first note
        global_end = song_data['metadata']['last_note']  # Ending tick of the last note
        time_signature_changes = song_data['metadata']['time_signature']  # Time signature changes
        ticks_per_beat = song_data['metadata']['ticks_per_beat']  # Ticks per beat resolution
        # Resolution for dividing beats within measures, expressed as a fraction
        in_beat_tick_resol = Fraction(ticks_per_beat, in_beat_resolution)  # Example: 1024/12 -> (256, 3)
        instrument_list = sorted(list(song_data['notes'].keys()))  # Get a sorted list of instruments in the song

        # --- process time signature --- #
        # Normalize and process the time signatures in the song
        time_signature_changes = self._process_time_signature(time_signature_changes, ticks_per_beat, first_note_tick, global_end)
        if time_signature_changes == None:
            return None  # Exit if time signature is invalid

        # --- create sequence --- #
        prev_instr_idx = None  # Track the previously processed instrument
        final_sequence = []
        final_sequence.append(self._create_event('SOS', None))  # Add Start of Sequence (SOS) token
        prev_chord = None  # Track the previous chord
        prev_tempo = None  # Track the previous tempo
        chord_value = None
        tempo_value = None

        # Process each time signature change
        for idx in range(len(time_signature_changes)):
            time_sig_change_flag = True  # Flag to indicate a time signature change
            # Calculate bar resolution based on the current time signature
            numerator = time_signature_changes[idx].numerator
            denominator = time_signature_changes[idx].denominator
            time_sig_name = f'time_signature_{numerator}/{denominator}'  # Format time signature name
            bar_resol = int(ticks_per_beat * numerator * (4 / denominator))  # Calculate bar resolution in ticks
            bar_start_tick = time_signature_changes[idx].time  # Start tick of the current bar
            # Determine the next time signature change point or the end of the song
            if idx == len(time_signature_changes) - 1:
                next_change_point = global_end
            else:
                next_change_point = time_signature_changes[idx+1].time
            
            # Process each measure within the current time signature
            for measure_step in frange(bar_start_tick, next_change_point, bar_resol):
                empty_bar_token = self._create_event('Bar', None)  # Token for empty bars

                # Ensure no more than 4 consecutive empty bars are added
                if len(final_sequence) >= 4:
                    if not (final_sequence[-1] == empty_bar_token and final_sequence[-2] == empty_bar_token and 
                            final_sequence[-3] == empty_bar_token and final_sequence[-4] == empty_bar_token):
                        if time_sig_change_flag:
                            final_sequence.append(self._create_event('Bar', time_sig_name))  # Mark new bar with time signature
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
                        
                time_sig_change_flag = False  # Reset time signature change flag
                
                # Process events within each beat
                for in_beat_off_idx, beat_step in enumerate(frange(measure_step, measure_step + bar_resol, in_beat_tick_resol)):
                    events_list = []
                    # Retrieve chords and tempos at the current beat step
                    t_chords = song_data['chords'].get(beat_step)
                    t_tempos = song_data['tempos'].get(beat_step)

                    # Process chord and tempo if the number of features allows for it
                    if self.num_features in {8, 7}:
                        if t_chords is not None:
                            root, quality, _ = t_chords[-1].text.split('_')  # Extract chord info
                            chord_value = root + '_' + quality
                        if t_tempos is not None:
                            tempo_value = t_tempos[-1].tempo  # Extract tempo value

                    # Dictionary to track notes for each instrument to avoid duplicates
                    instrument_note_dict = defaultdict(dict)

                    # Process notes for each instrument at the current beat step
                    for instrument_idx in instrument_list:
                        t_notes = song_data['notes'][instrument_idx].get(beat_step)

                        # If there are notes at this beat step, process them.
                        if t_notes is not None:
                            # Track notes to avoid duplicates and check for half-step intervals
                            for note in t_notes:
                                if note.pitch not in instrument_note_dict[instrument_idx]:
                                    instrument_note_dict[instrument_idx][note.pitch] = [(note.quantized_duration, note.velocity)]
                                else:
                                    instrument_note_dict[instrument_idx][note.pitch].append((note.quantized_duration, note.velocity))
                            
                    if len(instrument_note_dict) == 0:
                        continue

                    # Check for half-step interval gaps and handle them across instruments
                    pruned_instrument_note_dict = self._half_step_interval_gap_check_across_instruments(instrument_note_dict)

                    # add chord and tempo
                    if self.num_features in {7, 8}:
                        if prev_chord != chord_value:
                            events_list.append(self._create_event('Chord', chord_value))
                            prev_chord = chord_value
                        if prev_tempo != tempo_value:
                            events_list.append(self._create_event('Tempo', tempo_value))
                            prev_tempo = tempo_value

                    # add instrument and note
                    for instrument in pruned_instrument_note_dict:
                        if self.num_features in {5, 8}:
                            events_list.append(self._create_event('Instrument', instrument))
                        
                        for pitch in pruned_instrument_note_dict[instrument]:
                            max_duration = max(pruned_instrument_note_dict[instrument][pitch], key=lambda x: x[0])
                            note_event = [
                                self._create_event('Note_Pitch', pitch),
                                self._create_event('Note_Duration', max_duration[0])
                            ]
                            if self.num_features in {7, 8}:
                                note_event.append(self._create_event('Note_Velocity', max_duration[1]))
                            events_list.extend(note_event)

                    # If there are events in this step, add a "Beat" event and the collected events
                    if len(events_list):
                        final_sequence.append(self._create_event('Beat', in_beat_off_idx))
                        final_sequence.extend(events_list)

        # --- end with BAR & EOS --- #
        final_sequence.append(self._create_event('Bar', None))  # Add final bar token
        final_sequence.append(self._create_event('EOS', None))  # Add End of Sequence (EOS) token
        return final_sequence 

################################# for CP style encoding #################################
  
class Corpus2event_cp(Corpus2event_remi):
    def __init__(self, num_features):
        super().__init__(num_features)
        self.num_features = num_features
        self._init_event_template()
      
    def _init_event_template(self):
        '''
        The order of musical features is Type, Beat, Chord, Tempo, Instrument, Pitch, Duration, Velocity
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

    def create_cp_sos_event(self):
        total_event = self.event_template.copy()
        total_event['type'] = 'SOS'
        return total_event
      
    def create_cp_eos_event(self):
        total_event = self.event_template.copy()
        total_event['type'] = 'EOS'
        return total_event

    def create_cp_metrical_event(self, pos, chord, tempo):
        '''
        when the compound token is related to metrical information
        '''
        meter_event = self.event_template.copy()
        meter_event['type'] = 'Metrical'
        meter_event['beat'] = pos
        if self.num_features == 7 or self.num_features == 8:
            meter_event['chord'] = chord
            meter_event['tempo'] = tempo
        return meter_event

    def create_cp_note_event(self, instrument_name, pitch, duration, velocity):
        '''
        when the compound token is related to note information
        '''
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
        # --- global tag --- #
        first_note_tick = song_data['metadata']['first_note']  # First note timestamp in ticks
        global_end = song_data['metadata']['last_note']  # Last note timestamp in ticks
        time_signature_changes = song_data['metadata']['time_signature']  # Time signature changes throughout the song
        ticks_per_beat = song_data['metadata']['ticks_per_beat']  # Ticks per beat (resolution of the timing grid)
        in_beat_tick_resol = Fraction(ticks_per_beat, in_beat_resolution)  # Tick resolution for beats
        instrument_list = sorted(list(song_data['notes'].keys()))  # List of instruments in the song

        # --- process time signature --- #
        # Process time signature changes and adjust them for the given song structure
        time_signature_changes = self._process_time_signature(time_signature_changes, ticks_per_beat, first_note_tick, global_end)
        if time_signature_changes == None:
            return None  # Exit if no valid time signature changes found

        # --- create sequence --- #
        final_sequence = []  # Initialize the final sequence to store the events
        final_sequence.append(self.create_cp_sos_event())  # Add the Start-of-Sequence (SOS) event
        chord_text = None  # Placeholder for the current chord
        tempo_text = None  # Placeholder for the current tempo

        # Loop through each time signature change and process the corresponding measures
        for idx in range(len(time_signature_changes)):
            time_sig_change_flag = True  # Flag to track when time signature changes
            # Calculate bar resolution (number of ticks per bar based on the time signature)
            numerator = time_signature_changes[idx].numerator
            denominator = time_signature_changes[idx].denominator
            time_sig_name = f'time_signature_{numerator}/{denominator}'  # Format the time signature as a string
            bar_resol = int(ticks_per_beat * numerator * (4 / denominator))  # Calculate number of ticks per bar
            bar_start_tick = time_signature_changes[idx].time  # Starting tick for this time signature

            # Determine the point for the next time signature change or the end of the song
            if idx == len(time_signature_changes) - 1:
                next_change_point = global_end
            else:
                next_change_point = time_signature_changes[idx + 1].time

            # Iterate over each measure (bar) between the current and next time signature change
            for measure_step in frange(bar_start_tick, next_change_point, bar_resol):
                empty_bar_token = self.create_cp_bar_event()  # Create an empty bar event

                # Check if the last four events in the sequence are consecutive empty bars
                if len(final_sequence) >= 4:
                    if not (final_sequence[-1] == empty_bar_token and final_sequence[-2] == empty_bar_token and final_sequence[-3] == empty_bar_token and final_sequence[-4] == empty_bar_token):
                        final_sequence.append(self.create_cp_bar_event(time_sig_change_flag, time_sig_name))
                    else:
                        if time_sig_change_flag:
                            final_sequence.append(self.create_cp_bar_event(time_sig_change_flag, time_sig_name))
                else:
                    final_sequence.append(self.create_cp_bar_event(time_sig_change_flag, time_sig_name))
                
                # Reset the time signature change flag after handling the bar event
                time_sig_change_flag = False

                # Loop through beats in each measure based on the in-beat resolution
                for in_beat_off_idx, beat_step in enumerate(frange(measure_step, measure_step + bar_resol, in_beat_tick_resol)):
                    chord_tempo_flag = False  # Flag to track if chord and tempo events are added
                    events_list = []  # List to hold events for the current beat
                    pos_text = 'Beat_' + str(in_beat_off_idx)  # Create a beat event label

                    # --- chord & tempo processing --- #
                    # Unpack chords and tempos for the current beat step
                    t_chords = song_data['chords'].get(beat_step)
                    t_tempos = song_data['tempos'].get(beat_step)

                    # If a chord is present, extract its root, quality, and bass
                    if self.num_features in {7, 8}:
                        if t_chords is not None:
                            root, quality, _ = t_chords[-1].text.split('_')
                            chord_text = 'Chord_' + root + '_' + quality

                        # If a tempo is present, format it as a string
                        if t_tempos is not None:
                            tempo_text = 'Tempo_' + str(t_tempos[-1].tempo)
                    
                    # Dictionary to track notes for each instrument to avoid duplicates
                    instrument_note_dict = defaultdict(dict)

                    # --- instrument & note processing --- #
                    # Loop through each instrument and process its notes at the current beat step
                    for instrument_idx in instrument_list:
                        t_notes = song_data['notes'][instrument_idx].get(beat_step)

                        # If notes are present, process them
                        if t_notes != None:
                            # Track notes and their properties (duration and velocity) for the current instrument
                            for note in t_notes:
                                if note.pitch not in instrument_note_dict[instrument_idx]:
                                    instrument_note_dict[instrument_idx][note.pitch] = [(note.quantized_duration, note.velocity)]
                                else:
                                    instrument_note_dict[instrument_idx][note.pitch].append((note.quantized_duration, note.velocity))

                    if len(instrument_note_dict) == 0:
                        continue
                    
                    # Check for half-step interval gaps and handle them across instruments
                    pruned_instrument_note_dict = self._half_step_interval_gap_check_across_instruments(instrument_note_dict)

                    # add chord and tempo
                    if self.num_features in {7, 8}:
                        if not chord_tempo_flag:
                            if chord_text == None:
                                chord_text = 'Chord_N_N'
                            if tempo_text == None:
                                tempo_text = 'Tempo_N_N'
                            events_list.append(self.create_cp_metrical_event(pos_text, chord_text, tempo_text))
                            chord_tempo_flag = True

                    # add instrument and note
                    for instrument_idx in pruned_instrument_note_dict:
                        instrument_name = 'Instrument_' + str(instrument_idx)
                        for pitch in pruned_instrument_note_dict[instrument_idx]:
                            max_duration = max(pruned_instrument_note_dict[instrument_idx][pitch], key=lambda x: x[0])
                            note_pitch_text = 'Note_Pitch_' + str(pitch)
                            note_duration_text = 'Note_Duration_' + str(max_duration[0])
                            note_velocity_text = 'Note_Velocity_' + str(max_duration[1])
                            events_list.append(self.create_cp_note_event(instrument_name, note_pitch_text, note_duration_text, note_velocity_text))                   

                    # If there are any events for this beat, add them to the final sequence
                    if len(events_list) > 0:
                      final_sequence.extend(events_list)

        # --- end with BAR & EOS --- #
        final_sequence.append(self.create_cp_bar_event())  # Add the final bar event
        final_sequence.append(self.create_cp_eos_event())  # Add the End-of-Sequence (EOS) event
        return final_sequence  # Return the final sequence of events
  
################################# for NB style encoding #################################

class Corpus2event_nb(Corpus2event_cp):
    def __init__(self, num_features):
        '''
        For convenience in logging, we use "type" word for "metric" sub-token in the code to compare easily with other encoding schemes
        '''
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

    def create_nb_sos_event(self):
        total_event = self.event_template.copy()
        total_event['type'] = 'SOS'
        return total_event

    def create_nb_eos_event(self):
        total_event = self.event_template.copy()
        total_event['type'] = 'EOS'
        return total_event

    def create_nb_event(self, bar_beat_type, pos, chord, tempo, instrument_name, pitch, duration, velocity):
        total_event = self.event_template.copy()
        total_event['type'] = bar_beat_type
        total_event['beat'] = pos
        total_event['pitch'] = pitch
        total_event['duration'] = duration
        if self.num_features in {5, 8}:
            total_event['instrument'] = instrument_name
        if self.num_features in {7, 8}:
            total_event['chord'] = chord
            total_event['tempo'] = tempo
            total_event['velocity'] = velocity
        return total_event

    def create_nb_empty_bar_event(self):
        total_event = self.event_template.copy()
        total_event['type'] = 'Empty_Bar'
        return total_event

    def get_bar_beat_idx(self, bar_flag, beat_flag, time_sig_name, time_sig_change_flag):
        '''
        This function is to get the metric information for the current bar and beat
        There are four types of metric information: NNN, SNN, SSN, SSS
        Each letter represents the change of time signature, bar, and beat (new or same)
        '''
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
        # --- global tag --- #
        first_note_tick = song_data['metadata']['first_note']  # First note timestamp in ticks
        global_end = song_data['metadata']['last_note']  # Last note timestamp in ticks
        time_signature_changes = song_data['metadata']['time_signature']  # Time signature changes throughout the song
        ticks_per_beat = song_data['metadata']['ticks_per_beat']  # Ticks per beat (resolution of the timing grid)
        in_beat_tick_resol = Fraction(ticks_per_beat, in_beat_resolution)  # Tick resolution for beats
        instrument_list = sorted(list(song_data['notes'].keys()))  # List of instruments in the song

        # --- process time signature --- #
        # Process time signature changes and adjust them for the given song structure
        time_signature_changes = self._process_time_signature(time_signature_changes, ticks_per_beat, first_note_tick, global_end)
        if time_signature_changes == None:
            return None  # Exit if no valid time signature changes found
          
        # --- create sequence --- #
        final_sequence = []  # Initialize the final sequence to store the events
        final_sequence.append(self.create_nb_sos_event())  # Add the Start-of-Sequence (SOS) event
        chord_text = None  # Placeholder for the current chord
        tempo_text = None  # Placeholder for the current tempo

        # Loop through each time signature change and process the corresponding measures
        for idx in range(len(time_signature_changes)):
            time_sig_change_flag = True  # Flag to track when time signature changes
            # Calculate bar resolution (number of ticks per bar based on the time signature)
            numerator = time_signature_changes[idx].numerator
            denominator = time_signature_changes[idx].denominator
            time_sig_name = f'time_signature_{numerator}/{denominator}'  # Format the time signature as a string
            bar_resol = int(ticks_per_beat * numerator * (4 / denominator))  # Calculate number of ticks per bar
            bar_start_tick = time_signature_changes[idx].time  # Starting tick for this time signature

            # Determine the point for the next time signature change or the end of the song
            if idx == len(time_signature_changes) - 1:
                next_change_point = global_end
            else:
                next_change_point = time_signature_changes[idx + 1].time

            # Iterate over each measure (bar) between the current and next time signature change
            for measure_step in frange(bar_start_tick, next_change_point, bar_resol):
                bar_flag = True
                note_flag = False

                # Loop through beats in each measure based on the in-beat resolution
                for in_beat_off_idx, beat_step in enumerate(frange(measure_step, measure_step + bar_resol, in_beat_tick_resol)):
                    beat_flag = True
                    events_list = []
                    pos_text = 'Beat_' + str(in_beat_off_idx)

                    # --- chord & tempo processing --- #
                    # Unpack chords and tempos for the current beat step
                    t_chords = song_data['chords'].get(beat_step)
                    t_tempos = song_data['tempos'].get(beat_step)

                    # If a chord is present, extract its root, quality, and bass
                    if self.num_features == 8 or self.num_features == 7:
                        if t_chords is not None:
                            root, quality, _ = t_chords[-1].text.split('_')
                            chord_text = 'Chord_' + root + '_' + quality

                        # If a tempo is present, format it as a string
                        if t_tempos is not None:
                            tempo_text = 'Tempo_' + str(t_tempos[-1].tempo)

                    # Dictionary to track notes for each instrument to avoid duplicates
                    instrument_note_dict = defaultdict(dict)

                    # --- instrument & note processing --- #
                    # Loop through each instrument and process its notes at the current beat step
                    for instrument_idx in instrument_list:
                        t_notes = song_data['notes'][instrument_idx].get(beat_step)

                        # If notes are present, process them
                        if t_notes != None:
                          note_flag = True

                          # Track notes and their properties (duration and velocity) for the current instrument
                          for note in t_notes:
                              if note.pitch not in instrument_note_dict[instrument_idx]:
                                  instrument_note_dict[instrument_idx][note.pitch] = [(note.quantized_duration, note.velocity)]
                              else:
                                  instrument_note_dict[instrument_idx][note.pitch].append((note.quantized_duration, note.velocity))

                          # # Check for half-step interval gaps and handle them accordingly
                          # self._half_step_interval_gap_check(instrument_note_dict, instrument_idx)

                    if len(instrument_note_dict) == 0:
                        continue
                    
                    # Check for half-step interval gaps and handle them across instruments
                    pruned_instrument_note_dict = self._half_step_interval_gap_check_across_instruments(instrument_note_dict)

                    # add chord and tempo
                    if self.num_features in {7, 8}:
                        if chord_text == None:
                            chord_text = 'Chord_N_N'
                        if tempo_text == None:
                            tempo_text = 'Tempo_N_N'

                    # add instrument and note
                    for instrument_idx in pruned_instrument_note_dict:
                        instrument_name = 'Instrument_' + str(instrument_idx)
                        for pitch in pruned_instrument_note_dict[instrument_idx]:
                            max_duration = max(pruned_instrument_note_dict[instrument_idx][pitch], key=lambda x: x[0])
                            note_pitch_text = 'Note_Pitch_' + str(pitch)
                            note_duration_text = 'Note_Duration_' + str(max_duration[0])
                            note_velocity_text = 'Note_Velocity_' + str(max_duration[1])
                            bar_beat_type = self.get_bar_beat_idx(bar_flag, beat_flag, time_sig_name, time_sig_change_flag)
                            events_list.append(self.create_nb_event(bar_beat_type, pos_text, chord_text, tempo_text, instrument_name, note_pitch_text, note_duration_text, note_velocity_text))
                            bar_flag = False
                            beat_flag = False
                            time_sig_change_flag = False

                    # If there are any events for this beat, add them to the final sequence
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

        # --- end with BAR & EOS --- #
        final_sequence.append(self.create_nb_eos_event())
        return final_sequence