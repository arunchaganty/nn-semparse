"""Some code to deal with regex data."""
import collections
import glob
import json
import os
import re
import sys

IN_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'data/regex/json')
OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'data/regex/processed')

def split_regex(r):
  r = r[9:-2]
  replacements = [
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
  ]
  for a, b in replacements:
    r = r.replace(a, b)
  r = ' '.join(r.split())
  print r
  return r

def process(filename):
  print >> sys.stderr, 'Processing %s' % filename
  basename = os.path.basename(filename)
  stage = basename.split('.')[2]

  with open(filename) as f:
    in_data = json.load(f)

  out_data = []
  for ex in in_data:
    utterance = ex['utterance'].strip()
    regex = ex['targetValue']
    y = split_regex(regex)
    out_data.append((utterance, y))

  out_basename = 'regex_%s.tsv' % stage
  with open(os.path.join(OUT_DIR, out_basename), 'w') as f:
    for x, y in out_data:
      print >> f, '%s\t%s' % (x, y)


def main():
  if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)
  for filename in sorted(glob.glob(os.path.join(IN_DIR, '*.json'))):
    process(filename)


if __name__ == '__main__':
  main()
