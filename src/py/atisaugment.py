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
    for r_type in replacements:
      for x_rep, y_rep in replacements[r_type]:
        if x_rep in x and y_rep in y:
          x_template = x.replace(x_rep, '%(w)s')
          y_template = y.replace(y_rep, '%(w)s')
          templates.append((x_template, y_template, r_type))
  for k in replacements:
    num_templates = sum(1 for x, y, t in templates if t == k)
    print >> sys.stderr, 'Found %d templates of type %s' % (num_templates, k)
  return templates

def augment_data(in_data):
  replacements = get_replacements()
  templates = get_templates(in_data, replacements)
  augmented_data = collections.defaultdict(list)
  for x_template, y_template, r_type in templates:
    cur_reps = replacements[r_type]
    for x_rep, y_rep in replacements[r_type]:
      augmented_data[r_type].append(
          (x_template % {'w': x_rep}, y_template % {'w': y_rep}))

  for z in augmented_data:
    print >> sys.stderr, 'Generated %d new %s examples' % (len(augmented_data[z]), z)
  return [(x, y) for z in augmented_data for x, y in augmented_data[z]]

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
  augmented_data = augment_data(in_data)

  # Write a sample of augmented data
  sampled_data = random.sample(augmented_data, 2000)
  write_data('atis_train_augment2k.tsv', in_data + sampled_data)

  # Write a sample of augmented data
  sampled_data = random.sample(augmented_data, 4000)
  write_data('atis_train_augment4k.tsv', in_data + sampled_data)

def main():
  process(IN_FILE)

if __name__ == '__main__':
  main()
