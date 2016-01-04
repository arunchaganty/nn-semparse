"""Augment regex data.

Very simple, just replace things in quotes and numbers.
"""
import os
import random
import re
import sys

IN_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'data/regex/processed/regex_train_sm.tsv')
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
  replacements = set((s, s[2:-2].replace(' ', ' _ '))
                     for x, y in in_data for s in get_quoted_strs(x))
  print >> sys.stderr, 'Found %d distinct quoted strings' % len(replacements)
  return replacements

def get_templates(in_data):
  str_templates = set()
  int_templates = set()
  for x, y in in_data:
    x_toks = x.split()
    y_toks = y.split()

    # Handle quoted stuff
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

    # Handle ints
    # For high precision, only handle case where both x and y have 1 int
    x_ints = [i for i in range(len(x_toks)) if x_toks[i].isdigit()]
    y_ints = [i for i in range(len(y_toks)) if y_toks[i].isdigit()]
    if len(x_ints) == 1 and len(y_ints) == 1:
      x_ind = x_ints[0]
      y_ind = y_ints[0]
      diff = int(y_toks[y_ind]) - int(x_toks[x_ind])
      x_new = ' '.join(swap_in(x_toks, x_ind, '%d'))
      y_new = ' '.join(swap_in(y_toks, y_ind, '%d'))
      int_templates.add((x_new, y_new, diff))

  #for x, y, n in sorted(list(str_templates)):
  #  print (x, y, n)

  print >> sys.stderr, 'Extracted %d string templates' % len(str_templates)  
  print >> sys.stderr, 'Extracted %d int templates' % len(int_templates)
  return str_templates, int_templates

def augment_data(in_data, num_str=0, num_int=0):
  """Align based on words in quotes and numbers."""

  # Create unknown strings
  def new_unk_replacement():
    s = 'synth:%03d' % new_unk_replacement.cur_unk
    new_unk_replacement.cur_unk += 1
    return (s, s)
  new_unk_replacement.cur_unk = 0

  # Strings
  str_augmented_data = set()
  replacements = get_replacements(in_data)
  str_templates, int_templates = get_templates(in_data)
  while len(str_augmented_data) < num_str:
    x_t, y_t, n = random.sample(str_templates, 1)[0]
    cur_reps = random.sample(replacements, n)
    x_reps = dict(('w%d' % i, cur_reps[i][0]) for i in range(n))
    y_reps = dict(('w%d' % i, cur_reps[i][1]) for i in range(n))
    x_new = x_t % x_reps
    y_new = y_t % y_reps
    str_augmented_data.add((x_new, y_new))
  str_augmented_data = list(str_augmented_data)
 
  # Ints
  int_augmented_data = []
  for x_template, y_template, diff in int_templates:
    for i in INTS:
      int_augmented_data.append((x_template % i, y_template % (i + diff)))
  random.shuffle(int_augmented_data)
  int_augmented_data = int_augmented_data[:num_int]

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
    basename = 'regex_train_sm_%dstr_%dint.tsv' % (len(cur_str), len(cur_int))
    write_data(basename, in_data + cur_str + cur_int)

  for i in (0, 100, 200, 300):
    for j in (0, 100, 200):
      write_subset(num_str=i, num_int=j)

def main():
  process(IN_FILE)

if __name__ == '__main__':
  main()
