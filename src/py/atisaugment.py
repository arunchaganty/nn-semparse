"""Augment ATIS data.

Use database, replace cities, airports, and airlines 
"""
import collections
import csv
import glob
import os
import random
import re
import sys

import atislexicon

IN_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'data/atis/processed/atis_train.tsv')
DB_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'data/atis/db')
OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'data/atis/processed-augmented')

def get_templates_and_replacements(data):
  lex = atislexicon.get_lexicon()
  templates = []
  replacements = collections.defaultdict(set)

  for x, y in data:
    x_toks = x.split(' ')
    y_toks = y.split(' ')
    lex_items = lex.map_over_sentence(x_toks, return_entries=True)
    lex_ents = [x[1] for x in lex_items]
    x_holes = []
    y_holes = []
    reptypes = []
    for (i, j), ent in lex_items:
      # Make sure this entity occurs exactly once in lexicon entries
      # and in the logical form
      if lex_ents.count(ent) != 1: continue
      if y_toks.count(ent) != 1: continue

      # Add the replacement rule
      x_span = ' '.join(x_toks[i:j])
      ent_type = ent.split(':')[1]
      replacements[ent_type].add((x_span, ent))

      # Update the template
      x_holes.append((i, j))
      y_holes.append(y_toks.index(ent))
      reptypes.append(ent_type)

    # Generate the template
    if len(x_holes) == 0: continue
    x_new_toks = list(x_toks)
    y_new_toks = list(y_toks)
    for count, ((i, j), y_ind) in enumerate(zip(x_holes, y_holes)):
      fmt_str = '%(w' + str(count) + ')s'
      x_new_toks[i] = fmt_str
      for k in range(i+1, j):
        x_new_toks[k] = None
      y_new_toks[y_ind] = fmt_str
    x_t = ' '.join(t for t in x_new_toks if t is not None)
    y_t = ' '.join(y_new_toks)
    templates.append((x_t, y_t, reptypes))

  # Print results
#  for t in replacements:
#    print '%s:' % t
#    for x in replacements[t]:
#      print '  %s' % str(x)
#  for x_t, y_t, reps in templates:
#    print '%s -> %s (%s)' % (x_t, y_t, reps)

  return templates, replacements

def sample_sentence(templates, replacements):
  x_t, y_t, replist = random.sample(templates, 1)[0]
  cur_reps = [random.sample(replacements[t], 1)[0] for t in replist]
  x_reps = dict(('w%d' % i, cur_reps[i][0]) for i in range(len(replist)))
  y_reps = dict(('w%d' % i, cur_reps[i][1]) for i in range(len(replist)))
  x_new = x_t % x_reps
  y_new = y_t % y_reps
  return (x_new, y_new)

def augment_single(templates, replacements, num):
  aug_data = set()
  while len(aug_data) < num:
    x, y = sample_sentence(templates, replacements)
    aug_data.add((x, y))
  return list(aug_data)

def augment_double(templates, replacements, num):
  """For now, just do a concatenation."""
  aug_data = set()
  while len(aug_data) < num:
    x1, y1 = sample_sentence(templates, replacements)
    x2, y2 = sample_sentence(templates, replacements)
    x_new = '%s <sep> %s' % (x1, x2)
    y_new = '%s <sep> %s' % (y1, y2)
    aug_data.add((x_new, y_new))
  return list(aug_data)

def main():
  print 'main() does nothing right now'
  pass

if __name__ == '__main__':
  main()
