"""Artificial data."""
import collections
import os
import random
import re
import sys

VERSION = 1  # Increment when updating
OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'data/artificial')

ENTITIES = ['ent:%02d' % x for x in range(30)]
RELATIONS = ['rel:%02d' % x for x in range(30)]

def gen_base():
  entities = [(e, '_' + e) for e in ENTITIES]
  relations = [(r, '_' + r) for r in RELATIONS]
  return entities + relations

def gen_simple():
  data = []
  for e in ENTITIES:
    for r in RELATIONS:
      x = '%s of %s' % (r, e)
      y = '( _%s _%s )' % (r, e)
      data.append((x, y))
  random.shuffle(data)
  return data

def gen_nested():
  data = []
  for e in ENTITIES:
    for r1 in RELATIONS:
      for r2 in RELATIONS:
        x = '%s of %s of %s' % (r1, r2, e)
        y = '( _%s ( _%s _%s ) )' % (r1, r2, e)
        data.append((x, y))
  random.shuffle(data)
  return data

def gen_union():
  data = []
  for e1 in ENTITIES:
    for e2 in ENTITIES:
      for r in RELATIONS:
        x = '%s of %s or %s' % (r, e1, e2) 
        y = '( _%s ( union _%s _%s ) )' % (r, e1, e2)
        data.append((x, y))
  random.shuffle(data)
  return data


def write_data(basename, data):
  print >> sys.stderr, 'Writing %s' % basename
  out_path = os.path.join(OUT_DIR, basename)
  with open(out_path, 'w') as f:
    for x, y in data:
      print >>f, '%s\t%s' % (x, y)

def main():
  random.seed(0)
  base = gen_base()
  simple = gen_simple()
  nested = gen_nested()
  union = gen_union()
  simple_train, simple_test = simple[:300], simple[300:800]
  nested_train, nested_test = nested[:500], nested[500:1000]
  union_train, union_test = union[:500], union[500:1000]

  def write_train(dirname, **kwargs):
    sets = collections.OrderedDict([
        ('num_simple', simple),
        ('num_nested', nested_train),
        ('num_union', union_train),
    ])
    name = '_'.join('%s%03d' % (k[4:], num) for k, num in kwargs.iteritems())
    basename = 'train_%s.tsv' % name
    #data_lists = [base] + [sets[k][:num] for k, num in kwargs.iteritems()]
    data_lists = [sets[k][:num] for k, num in kwargs.iteritems()]
    data = [x for z in data_lists for x in z]
    write_data(os.path.join(dirname, basename), data)

  write_data('simple/simple_test500.tsv', simple_test)
  write_data('nested/nested_test500.tsv', nested_test)
  write_data('union/union_test500.tsv', union_test)

  write_train('nested', num_nested=100)
  for i in (25, 50, 75, 100, 150, 200, 250, 300):
    write_train('nested', num_nested=100, num_simple=i)
    write_train('nested', num_nested=100, num_union=i)
    write_train('nested', num_nested=100+i)

  write_train('simple', num_simple=100)
  for i in (25, 50, 75, 100, 150, 200, 250, 300):
    write_train('simple', num_simple=100, num_nested=i)
    write_train('simple', num_simple=100, num_union=i)
    write_train('simple', num_simple=100+i)

  write_train('union', num_union=100)
  for i in (25, 50, 75, 100, 150, 200, 250, 300):
    write_train('union', num_union=100, num_nested=i)
    write_train('union', num_union=100, num_simple=i)
    write_train('union', num_union=100+i)

if __name__ == '__main__':
  main()
