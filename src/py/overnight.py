"""Some code to deal with overnight data."""
import collections
import glob
import os
import sys

import sdf

IN_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'data/overnight/sdf')
GROUPS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'data/overnight/groups')
OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'data/overnight/processed')

def read_groups(filename):
  """Read a map from utterance to canonical utterance."""
  print >> sys.stderr, 'Reading groups for %s' % filename
  ret = {}
  canonical_utterance = None
  with open(filename) as f:
    for line in f:
      if line.startswith('original - '):
        canonical_utterance = line.split('-')[1].strip()
      elif line.startswith('  para - '):
        utterance = line.split('-')[1].split(',')[0].strip()
        if utterance in ret and ret[utterance] != canonical_utterance:
          print 'Collision: "%s"' % utterance
          print '  %s' % ret[utterance]
          print '  %s' % canonical_utterance
        # Prefer shorter canonical utterances
        if (utterance not in ret or
            len(canonical_utterance.split()) < len(ret[utterance].split())):
          ret[utterance] = canonical_utterance
  return ret

def process(filename, all_groups):
  print >> sys.stderr, 'Processing %s' % filename
  basename = os.path.basename(filename)
  domain = basename.split('_')[0]
  stage = basename.split('_')[1].split('.')[0]
  groups = all_groups[domain]

  in_data = sdf.read(filename)
  out_data = []
  for records in in_data:
    utterance = records[0].utterance.strip()
    true_canonical = groups[utterance]
    canonical_utterances = [r.canonical_utterance for r in records
                            if r.compatibility]
    out_data.append((utterance, true_canonical))
    if stage != 'train':
      # Write all correct canonical utterances
      for c in canonical_utterances:
        if c != true_canonical:
          out_data.append((utterance, c))

  out_basename = '%s_%s.tsv' % (domain, stage)
  with open(os.path.join(OUT_DIR, out_basename), 'w') as f:
    for ex in out_data:
      u, c = ex
      print >> f, '%s\t%s' % (u, c)

def concat_all(stage):
  with open(os.path.join(OUT_DIR, 'all_%s.tsv' % stage), 'w') as f_out:
    for filename in sorted(glob.glob(os.path.join(OUT_DIR, '*_%s.tsv' % stage))):
      if filename == 'all_%s.tsv' % stage: continue
      with open(filename) as f_in:
        f_out.write(''.join(f_in))

def main():
  all_groups = {}
  for filename in sorted(glob.glob(os.path.join(GROUPS_DIR, '*.groups'))):
    domain = os.path.basename(filename).split('.')[0]
    all_groups[domain] = read_groups(filename)
  for filename in sorted(glob.glob(os.path.join(IN_DIR, '*.sdf'))):
    process(filename, all_groups)

  # Create an all_train.tsv and all_dev.tsv file
  concat_all('train')
  concat_all('dev')

if __name__ == '__main__':
  main()
