"""Some code to deal with geo880 data."""
import collections
import glob
import os
import re
import sys

IN_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'data/geo880/sempre-examples')
OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'data/geo880/processed')

def read_examples(filename):
  examples = []
  utterance = None
  logical_form = None
  with open(filename) as f:
    for line in f:
      line = line.strip()
      if line.startswith('(utterance'):
        utterance = re.match('\(utterance "(.*)"\)', line).group(1)
      elif line.startswith('(targetFormula'):
        logical_form = re.match(
            r'\(targetFormula \(string "(.*)"\)\)', line).group(1)
        examples.append((utterance, logical_form))
  return examples

def split_logical_form(lf):
  replacements = [
      ('(', ' ( '),
      (')', ' ) '),
      (',', ' , '),
  ]
  for a, b in replacements:
    lf = lf.replace(a, b)
  return ' '.join(lf.split())

def process(filename):
  print >> sys.stderr, 'Processing %s' % filename
  basename = os.path.basename(filename)
  domain = basename.split('_')[0]
  stage = basename.split('_')[1].split('.')[0]
  
  in_data = read_examples(filename)
  out_data = []
  for (utterance, logical_form) in in_data:
    y = split_logical_form(logical_form)
    out_data.append((utterance, y))

  out_basename = '%s_%s.tsv' % (domain, stage)
  with open(os.path.join(OUT_DIR, out_basename), 'w') as f:
    for x, y in out_data:
      print >> f, '%s\t%s' % (x, y)

def main():
  if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)
  for filename in sorted(glob.glob(os.path.join(IN_DIR, '*.examples'))):
    process(filename)

if __name__ == '__main__':
  main()
