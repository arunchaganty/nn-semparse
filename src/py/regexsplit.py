"""Split regex into 3 folds."""
import os
import random
import sys

ALL_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'data/regex/processed/regex_train_all.tsv')
OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'data/regex/processed')

def read_examples(filename):
  with open(filename) as f:
    data = [tuple(line.strip().split('\t')) for line in f]
  print >> sys.stderr, 'Read %d examples' % len(data)
  return data

def write_data(basename, data):
  print >> sys.stderr, 'Writing %s' % basename
  out_path = os.path.join(OUT_DIR, basename)
  with open(out_path, 'w') as f:
    for x, y in data:
      print >>f, '%s\t%s' % (x, y)

def main():
  random.seed(314159)
  in_data = read_examples(ALL_FILE)
  random.shuffle(in_data)
  num_folds = 3
  fold_size = len(in_data) / num_folds
  folds = [in_data[fold_size*i:fold_size*(i+1)] for i in range(num_folds)]
  for i in range(num_folds):
    train_folds = [f for j, f in enumerate(folds) if j != i]
    cur_train = [x for f in train_folds for x in f]
    cur_test = folds[i]
    write_data('regex_fold%dof%d_train%d.tsv' % (i+1, num_folds, len(cur_train)), cur_train)
    write_data('regex_fold%dof%d_test%d.tsv' % (i+1, num_folds, len(cur_test)), cur_test)

if __name__ == '__main__':
  main()

