"""Augment regex data.

Very simple, just replace things in quotes and numbers.
"""
import collections
import os
import random
import re
import sys

from vocabulary import Vocabulary

IN_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'data/regex/processed')
IN_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'data/regex/processed/regex_train_sm.tsv')
ALL_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'data/regex/processed/regex_train_all.tsv')
OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'data/regex/processed-augmented')

def read_examples(filename):
  with open(filename) as f:
    data = [tuple(line.strip().split('\t')) for line in f]
  print >> sys.stderr, 'Read %d examples' % len(data)
  return data

INTS = range(1, 10)  # Use integers from 1 to 9 inclusive

def swap_in(old_list, index, new_val):
  """Return a new list where new_list[index] = new_val."""
  new_list = list(old_list)
  new_list[index] = new_val
  return new_list

def get_quoted_strs(x):
  return [m[0] for m in re.findall('((" [^"]* ")|(\' [^\']* \'))', x)]

def get_replacements(in_data):
  """Get all quoted strings actually used."""
  vocab = get_true_vocab(in_data, 1)
  quot_strs = set(s for x, y in in_data for s in get_quoted_strs(x))

  def mask_unk(s):
    """Replace rare words with UNK_%d markers."""
    toks = s.split(' ')
    new_toks = []
    unk_dict = {}
    for t in toks:
      if t in vocab:
        new_toks.append(t)
      else:
        if t not in unk_dict:
          unk_dict[t] = len(unk_dict)
        new_toks.append('UNK_%02d' % unk_dict[t])
    x_new = ' '.join(new_toks)
    y_new = x_new[2:-2].replace(' ', ' _ ')
    return (x_new, y_new)

  replacements = list(set([mask_unk(s) for s in quot_strs]))
  print >> sys.stderr, 'Found %d distinct quoted strings' % len(replacements)
  return replacements

def get_str_templates(in_data):
  str_templates = set()
  for x, y in in_data:
    x_toks = x.split()
    y_toks = y.split()

    quoted_strs = get_quoted_strs(x)
    if len(quoted_strs) == 0: continue 
    x_new = x
    y_new = y
    for i, x_quot in enumerate(quoted_strs):
      y_quot = x_quot[2:-2].replace(' ', ' _ ') 
      pattern = '(^| )%s($| )' % re.escape(y_quot)
      all_y_matches = re.findall(pattern, y)
      if len(all_y_matches) < 1:
        print >> sys.stderr, 'Quoted string "%s" found %d times in y = "%s"' % (
            y_quot, len(all_y_matches), y)
        x_new = None
        y_new = None
        break
      replacement_str = '%(w' + str(i) + ')s'
      x_new = x_new.replace(x_quot, replacement_str)
      y_new = re.sub(pattern, lambda m: m.group(1) + replacement_str + m.group(2), y_new)
    if x_new is None: continue
    str_templates.add((x_new, y_new, len(quoted_strs)))
  print >> sys.stderr, 'Extracted %d string templates' % len(str_templates)  
  return str_templates

def get_int_templates(in_data):
  int_templates = set()
  for x, y in in_data:
    x_toks = x.split()
    y_toks = y.split()
    x_ints = [i for i in range(len(x_toks)) if x_toks[i].isdigit()]
    y_ints = [i for i in range(len(y_toks)) if y_toks[i].isdigit()]
    if len(x_ints) == 1 and len(y_ints) == 1:
      x_ind = x_ints[0]
      y_ind = y_ints[0]
      diff = int(y_toks[y_ind]) - int(x_toks[x_ind])
      x_new = ' '.join(swap_in(x_toks, x_ind, '%d'))
      y_new = ' '.join(swap_in(y_toks, y_ind, '%d'))
      int_templates.add((x_new, y_new, diff))
  print >> sys.stderr, 'Extracted %d int templates' % len(int_templates)
  return int_templates

def get_true_vocab(in_data, unk_cutoff):
  sentences = [x for x, y in in_data]
  counts = collections.Counter()
  for s in sentences:
    counts.update(s.split(' '))
  vocab = set(w for w in counts if counts[w] > unk_cutoff)
  return vocab

def new_unk_replacement():
  s = 'synth:%04d' % new_unk_replacement.cur_unk
  new_unk_replacement.cur_unk += 1
  return s
new_unk_replacement.cur_unk = 0

def replace_all(s, d):
  for k, v in d.iteritems():
    s = s.replace(k, v)
  return s

def sample_one_str(str_templates, replacements):
  x_t, y_t, n = random.sample(str_templates, 1)[0]
  cur_reps = random.sample(replacements, n)
  for i in range(len(cur_reps)):
    x_r, y_r = cur_reps[i]
    unks = sorted(list(set(re.findall('UNK_[0-9]+', x_r))))
    unk_dict = dict((u, new_unk_replacement()) for u in unks)
    x_new = replace_all(x_r, unk_dict)
    y_new = replace_all(y_r, unk_dict)
    cur_reps[i] = (x_new, y_new)
  x_reps = dict(('w%d' % i, cur_reps[i][0]) for i in range(n))
  y_reps = dict(('w%d' % i, cur_reps[i][1]) for i in range(n))
  x_new = x_t % x_reps
  y_new = y_t % y_reps
  return (x_new, y_new)

def augment_str(in_data, str_templates, num):
  str_augmented_data = set()
  replacements = get_replacements(in_data)
  while len(str_augmented_data) < num:
    str_augmented_data.add(sample_one_str(str_templates, replacements))
  return list(str_augmented_data)
 
def augment_int(in_data, int_templates, num=None):
  int_augmented_data = []
  for x_template, y_template, diff in int_templates:
    for i in INTS:
      int_augmented_data.append((x_template % i, y_template % (i + diff)))
  random.shuffle(int_augmented_data)
  if num:
    return int_augmented_data[:num]
  else:
    return int_augmented_data

def augment_conj(in_data, str_templates, int_templates, num):
  all_int_data = augment_int(in_data, int_templates)
  replacements = get_replacements(in_data)

  def sample_sentence():
    if random.randint(0, len(str_templates) + len(int_templates)) <= len(str_templates):
      return sample_one_str(str_templates, replacements)
    else:
      return random.sample(all_int_data, 1)[0]

  def shorten(x):
    if x.startswith('lines'):
      toks = x.split(' ')
      if toks[1] in ('that', 'where', 'which', 'with'):
        return ' '.join(toks[1:])
      elif toks[1].endswith('ing'):
        toks[1] = toks[1][:-3]
        return 'that %s' % ' '.join(toks[1:])
    return 'that are %s' + x

  aug_data = set()
  while len(aug_data) < num:
    x1, y1 = sample_sentence()
    x2, y2 = sample_sentence()
    x_new = '%s %s' % (x1, shorten(x2))
    y_new = '( %s ) & ( %s )' % (y1, y2)
    aug_data.add((x_new, y_new))
  return list(aug_data)

def augment_data(in_data, num_str=0, num_int=0):
  """Align based on words in quotes and numbers."""
  str_templates = get_str_templates(in_data)
  int_templates = get_int_templates(in_data)
  str_augmented = augment_str(in_data, str_templates, num_str)
  int_augmented = augment_int(in_data, int_templates, num_int)
  return str_augmented_data, int_augmented_data

def write_data(basename, data):
  print >> sys.stderr, 'Writing %s' % basename
  out_path = os.path.join(OUT_DIR, basename)
  with open(out_path, 'w') as f:
    for x, y in data:
      print >>f, '%s\t%s' % (x, y)

def process(filename):
  random.seed(1)
  print >> sys.stderr, 'Processing %s' % filename
  basename = os.path.basename(filename)
  in_data = read_examples(filename)
  str_data, int_data = augment_data(in_data, num_str=1000, num_int=1000)

  def write_subset(num_str=0, num_int=0):
    cur_str = str_data[:num_str]
    cur_int = int_data[:num_int]
    if 'nkushman' in filename:
      # basename is like regex_train_nkushman_fold0.tsv
      mode = os.path.basename(filename)[12:-4]
    elif 'fold' in filename:
      # basename is like regex_fold1of3_train440.tsv
      mode = os.path.basename(filename)[6:-4]
    elif filename == IN_FILE:
      mode = 'sm'
    else:
      mode = 'all'
    basename = 'regex_train_%s_%dstr_%dint.tsv' % (mode, len(cur_str), len(cur_int))
    write_data(basename, in_data + cur_str + cur_int)

  for i in (0, 100, 200, 300):
    for j in (0, 100, 200):
      write_subset(num_str=i, num_int=j)

def main():
  process(IN_FILE)
  process(ALL_FILE)
  process(os.path.join(IN_DIR, 'regex_fold1of3_train440.tsv'))
  process(os.path.join(IN_DIR, 'regex_fold2of3_train440.tsv'))
  process(os.path.join(IN_DIR, 'regex_fold3of3_train440.tsv'))
  process(os.path.join(IN_DIR, 'regex_train_nkushman_fold0.tsv'))
  process(os.path.join(IN_DIR, 'regex_train_nkushman_fold1.tsv'))
  process(os.path.join(IN_DIR, 'regex_train_nkushman_fold2.tsv'))

if __name__ == '__main__':
  main()
