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
    'data/atis/processed-augmented')

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
  return s

def read_db(basename, id_col, name_col, id_suffix):
  filename = os.path.join(DB_DIR, basename)
  data = []  # Pairs of (name, id)
  with open(filename) as f:
    for line in f:
      row = [s[1:-1] for s in re.findall('"[^"]*"', line.strip())]
      cur_name = clean_name(row[name_col].lower())
      cur_id = clean_id(row[id_col].lower(), id_suffix)
      data.append((cur_name, cur_id))
  return data

def get_replacements():
  replacements = {}
  replacements['city'] = read_db('CITY.TAB', 1, 1, '_ci')
  replacements['airport'] = read_db('AIRPORT.TAB', 0, 1, '_ap')
  replacements['airline'] = read_db('AIRLINE.TAB', 0, 1, '_al')
  for k in replacements:
    print >> sys.stderr, 'Found %d replacements of type %s' % (len(replacements[k]), k)
  return replacements

def get_templates(in_data, replacements):
  templates = []
  for x, y in in_data:
    x_new = x
    y_new = y
    r_types = []
    for r_type in replacements:
      for x_rep, y_rep in replacements[r_type]:
        if x_rep in x and y_rep in y:
          x_new = x_new.replace(x_rep, '%(w' + str(len(r_types)) + ')s')
          y_new = y_new.replace(y_rep, '%(w' + str(len(r_types)) + ')s')
          r_types.append(r_type)
    if x_new != x:
      templates.append((x_new, y_new, r_types))
  return templates

def augment_data(in_data, num):
  replacements = get_replacements()
  templates = get_templates(in_data, replacements)
  augmented_data = set()

  while len(augmented_data) < num:
    template = random.sample(templates, 1)[0]
    x_t, y_t, r_types = template
    cur_reps = [random.sample(replacements[r_type], 1)[0] for r_type in r_types]
    x_reps = dict(('w%d' % i, cur_reps[i][0]) for i in range(len(r_types)))
    y_reps = dict(('w%d' % i, cur_reps[i][1]) for i in range(len(r_types)))
    x_new = x_t % x_reps
    y_new = y_t % y_reps
    augmented_data.add((x_new, y_new))
  return list(augmented_data)

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
  augmented_data = augment_data(in_data, 4000)

  write_data('atis_train_augment2k.tsv', in_data + augmented_data[:2000])
  write_data('atis_train_augment4k.tsv', in_data + augmented_data[:4000])

def main():
  process(IN_FILE)

if __name__ == '__main__':
  main()
