"""Some code to deal with regex data."""
import collections
import glob
import json
import os
import random
import re
import sys

IN_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'data/regex/json')
RAW_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'data/regex/raw')
OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'data/regex/processed')

random.seed(0)

def split_regex(r):
  r = r[9:-2]
  replacements = [
      (' ', ' _ '),
      ('(', ' ( '),
      (')', ' ) '),
      ('[', ' ['),
      (']', '] '),
      (r'\\b', r' \\b '),
      ('.', ' . '),
      ('*', ' * '),
      ('~', ' ~ '),
      ('&', ' & '),
      ('+', ' + '),
      ('{', ' { '),
      ('}', ' } '),
      ('|', ' | '),
      (',', ' , '),
  ]
  for a, b in replacements:
    r = r.replace(a, b)
  r = ' '.join(r.split())
  print r
  return r

def split_input(s):
  replacements = [
      ('"', ' " '),
      ("'", " ' "),
  ]
  for a, b in replacements:
    s = s.replace(a, b)
  s = ' '.join(s.split())
  return s

def process(filename):
  print >> sys.stderr, 'Processing %s' % filename
  basename = os.path.basename(filename)
  stage = basename.split('.')[2]

  with open(filename) as f:
    in_data = json.load(f)

  out_data = []
  for ex in in_data:
    utterance = ex['utterance'].strip()
    x = split_input(utterance)
    regex = ex['targetValue']
    y = split_regex(regex)
    out_data.append((x, y))

  if stage == 'train':
    out_basename = 'regex_%s_all.tsv' % stage
  else:
    out_basename = 'regex_%s.tsv' % stage
  with open(os.path.join(OUT_DIR, out_basename), 'w') as f:
    for x, y in out_data:
      print >> f, '%s\t%s' % (x, y)

  if stage == 'train':
    random.shuffle(out_data)
    train_sm_data = out_data[:-100]
    dev_data = out_data[-100:]
    with open(os.path.join(OUT_DIR, 'regex_train_sm.tsv'), 'w') as f:
      for x, y in train_sm_data:
        print >> f, '%s\t%s' % (x, y)
    with open(os.path.join(OUT_DIR, 'regex_dev.tsv'), 'w') as f:
      for x, y in dev_data:
        print >> f, '%s\t%s' % (x, y)


def write_data(basename, data):
  out_path = os.path.join(OUT_DIR, basename)
  with open(out_path, 'w') as f:
    for x, y in data:
      print >>f, '%s\t%s' % (x, y)

def process_raw(filename):
  print >> sys.stderr, 'Processing %s' % filename
  basename = os.path.basename(filename)
  # Filenames are like out_fold0.tsv
  fold = int(basename[8])

  train_data = []
  test_data = []
  with open(filename) as f:
    for line in f:
      stage, x_raw, y_raw = line.strip().split('\t')
      x = split_input(x_raw)
      y = split_regex(y_raw)
      if stage == 'train':
        train_data.append((x, y))
      else:
        test_data.append((x, y))

  train_file = 'regex_train_nkushman_fold%d.tsv' % fold
  test_file = 'regex_test_nkushman_fold%d.tsv' % fold
  write_data(train_file, train_data)
  write_data(test_file, test_data)

def main():
  if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)
  for filename in sorted(glob.glob(os.path.join(IN_DIR, '*.json'))):
    process(filename)
  for filename in sorted(glob.glob(os.path.join(RAW_DIR, '*.tsv'))):
    process_raw(filename)



if __name__ == '__main__':
  main()
