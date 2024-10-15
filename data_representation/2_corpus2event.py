import argparse
import time
from pathlib import Path

import pickle
from tqdm import tqdm

import encoding_utils

'''
This script is for converting corpus data to event data.
'''

class Corpus2Event():
  def __init__(
      self, 
      dataset: str, 
      encoding_scheme: str, 
      num_features: int, 
      in_dir: Path, 
      out_dir: Path,
      debug: bool
  ):
    self.dataset = dataset
    self.encoding_name = encoding_scheme + str(num_features)
    self.in_dir = in_dir / f"corpus_{self.dataset}"
    self.out_dir = out_dir / f"events_{self.dataset}" / self.encoding_name
    self.debug = debug
    self.encoding_function = getattr(encoding_utils, f'Corpus2event_{encoding_scheme}')(num_features)
    self._get_in_beat_resolution()

  def _get_in_beat_resolution(self):
    in_beat_resolution_dict = {'BachChorale': 4, 'Pop1k7': 4, 'Pop909': 4, 'SOD': 12, 'LakhClean': 4, 'SymphonyMIDI': 8}
    self.in_beat_resolution = in_beat_resolution_dict[self.dataset]

  def make_events(self):
    '''
    Preprocess corpus data to events data.
    The process in each encoding scheme is different.
    Please refer to encoding_utils.py for more details.
    '''
    print("preprocessing corpus data to events data")
    # check output directory exists
    self.out_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    # single-processing
    broken_count = 0
    success_count = 0
    corpus_list = sorted(list(self.in_dir.rglob("*.pkl")))
    for filepath_name, event in tqdm(map(self._load_single_corpus_and_make_event, corpus_list), total=len(corpus_list)):
      if event is None:
        broken_count += 1
        continue
      with open(self.out_dir / filepath_name, 'wb') as f:
        pickle.dump(event, f)
      success_count += 1
      del event
    print(f"taken time for making events is {time.time()-start_time}s, success: {success_count}, broken: {broken_count}")

  def _load_single_corpus_and_make_event(self, file_path):
    with open(file_path, 'rb') as f:
      corpus = pickle.load(f)
    event = self.encoding_function(corpus, self.in_beat_resolution)
    return file_path.name, event

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
      "-e",
      "--encoding",
      required=True,
      choices=("remi", "cp", "nb", "remi_pos"),
      type=str,
      help="encoding scheme",
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
      default="../dataset/represented_data/corpus/",
      type=Path,
      help="input data directory",
  )
  parser.add_argument(
      "-o",
      "--out_dir",
      default="../dataset/represented_data/events/",
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
  args = get_argument_parser().parse_args()
  corpus2event = Corpus2Event(args.dataset, args.encoding, args.num_features, args.in_dir, args.out_dir, args.debug)
  corpus2event.make_events()

if __name__ == "__main__":
  main()