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

SYNTHETIC_WORDS = ['synth:%02d' % i for i in range(100)]
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
    for x_quot in quoted_strs:
      y_quot = x_quot[2:-2].replace(' ', ' _ ') 
      pattern = '(^| )%s($| )' % re.escape(y_quot)
      if not re.search(pattern, y):
        print >> sys.stderr, 'Quoted string "%s" not found in y = "%s"' % (y_quot, y)
        continue
      x_new = x.replace(x_quot, '%(w)s')
      y_new = re.sub(pattern, lambda m: m.group(1) + '%(w)s' + m.group(2), y)
      str_templates.add((x_new, y_new))

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

  #for x, y in sorted(list(str_templates)):
  #  print (x, y)

  print >> sys.stderr, 'Extracted %d string templates' % len(str_templates)  
  print >> sys.stderr, 'Extracted %d int templates' % len(int_templates)
  return str_templates, int_templates

def augment_data(in_data):
  """Align based on words in quotes and numbers.
  
  To avoid combinatorial explosion, only swap in one thing at a time.
  """
  str_augmented_data = []
  int_augmented_data = []

  # Create unknown strings
  def new_unk_replacement():
    s = 'synth:%03d' % new_unk_replacement.cur_unk
    new_unk_replacement.cur_unk += 1
    return (s, s)
  new_unk_replacement.cur_unk = 0

  replacements = get_replacements(in_data)
  str_templates, int_templates = get_templates(in_data)

  for x_template, y_template in str_templates:
    #cur_replacements = replacements | set([new_unk_replacement()])
    cur_replacements = replacements
    for x_rep, y_rep in cur_replacements:
      str_augmented_data.append((x_template % {'w': x_rep},
                                 y_template % {'w': y_rep}))
 
  for x_template, y_template, diff in int_templates:
    for i in INTS:
      int_augmented_data.append((x_template % i, y_template % (i + diff)))

  print >> sys.stderr, 'Generated %d new str examples' % len(str_augmented_data)
  print >> sys.stderr, 'Generated %d new int examples' % len(int_augmented_data)
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
  str_data, int_data = augment_data(in_data)

  # Write everything
  write_data('regex_train_sm_augmentAll.tsv', in_data + str_data + int_data)

  # Write a sample of str data
  sampled_str = random.sample(str_data, 1000)
  write_data('regex_train_sm_augment1kStr.tsv', in_data + sampled_str)

  # Write int data only
  write_data('regex_train_sm_augmentInt.tsv', in_data + int_data)

  # Write sample of str data + int data
  write_data('regex_train_sm_augment1kStrPlusInt.tsv', in_data + sampled_str + int_data)

def main():
  process(IN_FILE)

if __name__ == '__main__':
  main()
