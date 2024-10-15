import argparse
from pathlib import Path

import vocab_utils

'''
This script is for creating vocab file.
'''

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
      choices=("remi", "cp", "nb"),
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
      default="../dataset/represented_data/events/",
      type=Path,
      help="input data directory",
  )
  parser.add_argument(
      "-o",
      "--out_dir",
      default="../vocab/",
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
  encoding_scheme = args.encoding
  num_features = args.num_features
  dataset = args.dataset

  out_vocab_path = args.out_dir / f"vocab_{dataset}"
  out_vocab_path.mkdir(parents=True, exist_ok=True)
  out_vocab_file_path = out_vocab_path / f"vocab_{dataset}_{encoding_scheme}{num_features}.json"
  
  events_path = Path(args.in_dir / f"events_{dataset}" / f"{encoding_scheme}{num_features}")
  vocab_name = {'remi':'LangTokenVocab', 'cp':'MusicTokenVocabCP', 'nb':'MusicTokenVocabNB'}
  selected_vocab_name = vocab_name[encoding_scheme]
  event_data = sorted(list(events_path.rglob("*.pkl")))
  vocab = getattr(vocab_utils, selected_vocab_name)(
    in_vocab_file_path=None, 
    event_data=event_data,
    encoding_scheme=encoding_scheme, 
    num_features=num_features
    )
  vocab.save_vocab(out_vocab_file_path)
  print(f"Vocab file saved at {out_vocab_file_path}")

if __name__ == "__main__":
  main()