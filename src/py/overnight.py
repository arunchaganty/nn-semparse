"""Map overnight to logical form."""
import collections
import glob
import os
import sys

import sdf

IN_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'data/overnight/paraphrases')
OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'data/overnight/processed-lf')

def split_lf(lf):
  replacements = [
      ('(', ' ( '),
      (')', ' ) '),
      ('!', ' ! '),
      ('edu.stanford.nlp.sempre.overnight.SimpleWorld', 'SW')  # For brevity
  ]
  for a, b in replacements:
    lf = lf.replace(a, b)
  return ' '.join(lf.split())


def process(filename):
  print >> sys.stderr, 'Processing %s' % filename
  basename = os.path.basename(filename)
  domain = basename.split('.')[0]
  stage = basename.split('.')[2]

  data = []
  x = None
  y = None
  next_is_lf = False
  with open(filename) as f:
    for line in f:
      if next_is_lf:
        y = line.strip()
        data.append((x, y))
        next_is_lf = False
      elif line.startswith('  (utterance'):
        x = line.split('"')[1]
      elif line.startswith('  (targetFormula'):
        next_is_lf = True

  out_basename = '%s_%s.tsv' % (domain, stage)
  with open(os.path.join(OUT_DIR, out_basename), 'w') as f:
    for ex in data:
      u, c = ex
      print >> f, '%s\t%s' % (u, c)

def concat_all(stage):
  with open(os.path.join(OUT_DIR, 'all_%s.tsv' % stage), 'w') as f_out:
    for filename in sorted(glob.glob(os.path.join(OUT_DIR, '*_%s.tsv' % stage))):
      if filename == 'all_%s.tsv' % stage: continue
      with open(filename) as f_in:
        f_out.write(''.join(f_in))

def main():
  for filename in sorted(glob.glob(os.path.join(IN_DIR, '*.examples'))):
    process(filename)

  # Create an all_train.tsv file
  concat_all('train')

if __name__ == '__main__':
  main()
