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

IN_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'data/atis/processed/atis_train.tsv')
DB_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'data/atis/db')
OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'data/atis/processed-lexicon')

def read_examples(filename):
  with open(filename) as f:
    data = [tuple(line.strip().split('\t')) for line in f]
  print 'Read %d examples' % len(data)
  return data

def clean_id(s, id_suffix):
  true_id = s.replace(' ', '_')
  return '%s : %s' % (true_id, id_suffix)

def clean_name(s):
  if s.endswith(', inc.') or s.endswith(', ltd.'): 
    s = s[:-6]
  s = s.replace('/', ' ')
  s = s.replace('.', '').replace('-', '').replace("'", '')
  return s

def read_db(basename, id_col, name_col, id_suffix):
  filename = os.path.join(DB_DIR, basename)
  data = []  # Pairs of (name, id)
  with open(filename) as f:
    for line in f:
      row = [s[1:-1] for s in re.findall('"[^"]*"|[0-9]+', line.strip())]
      cur_name = clean_name(row[name_col].lower())
      cur_id = clean_id(row[id_col].lower(), id_suffix)
      data.append((cur_name, cur_id))
  for ex in data:
    print ex
  return data

DAYS_OF_WEEK = [(s, '%s : _da' % s) for s in 
                ['monday', 'tuesday', 'wednesday', 'thursday', 'friday',
                 'saturday', 'sunday']]
DATE_NUMBERS = [('%d' % i, '%d : _dn' % i) for i in range(1, 32)]


def get_lexicon_items():
  lex_items = []
  lex_items.extend(read_db('CITY.TAB', 1, 1, '_ci'))
  lex_items.extend(DAYS_OF_WEEK)
  lex_items.extend(read_db('AIRLINE.TAB', 0, 1, '_al'))
  # Nothing good for times
  # Nothing good for time of day
  lex_items.extend(DAYS_OF_WEEK)
  lex_items.extend(read_db('MONTH.TAB', 1, 1, '_mn'))
  lex_items.extend(read_db('AIRPORT.TAB', 0, 1, '_ap'))
  lex_items.extend(read_db('COMP_CLS.TAB', 1, 1, '_cl'))

  print >> sys.stderr, 'Found %d lexicon items' % len(lex_items)
  return lex_items
#   8207 _ci = cities
#    888 _da = days of the week
#    735 _al = airlines
#    607 _ti = times
#    594 _pd = time of day
#    404 _dn = date number
#    389 _mn = month
#    203 _ap = airport
#    193 _cl = class
# ----------------------
#     72 _fb
#     65 _fn = flight number
#     52 _me = meal
#     50 _do
#     28 _rc
#     23 _ac
#     22 _yr
#      4 _mf
#      2 _dc
#      2 _st
#      1 _hr

def write_data(basename, data):
  out_path = os.path.join(OUT_DIR, basename)
  with open(out_path, 'w') as f:
    for x, y in data:
      print >>f, '%s\t%s' % (x, y)

def process(filename):
  random.seed(1)
  print >> sys.stderr, 'Processing %s' % filename
  basename = os.path.basename(filename)
  in_data = read_examples(filename)
  lexicon_items = get_lexicon_items()
  write_data('atis_train_augmentLex.tsv', in_data + lexicon_items)

def main():
  process(IN_FILE)

if __name__ == '__main__':
  main()
