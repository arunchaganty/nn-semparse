"""Artificial data."""
import collections
import itertools
import os
import random
import re
import sys

VERSION = 1  # Increment when updating
OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'data/artificial')

ENTITIES = ['ent:%02d' % x for x in range(20)]
RELATIONS = ['rel:%02d' % x for x in range(50)]

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

def gen_nested(depth=2):
  data = []
  for e in ENTITIES:
    for rels in itertools.product(RELATIONS, repeat=depth):
      rels = list(rels)
      x = ' of '.join(rels + [e])
      y = '( ' + ' ( '.join(['_%s' % r for r in rels]) + ' _%s' % e + ' )' * depth
      data.append((x, y))
  random.shuffle(data)
  return data

def sample_nested(depth=2, num=0):
  data = set()
  while len(data) < num:
    rels = [random.sample(RELATIONS, 1)[0] for i in range(depth)]
    e = random.sample(ENTITIES, 1)[0]
    x = ' of '.join(rels + [e])
    y = '( ' + ' ( '.join(['_%s' % r for r in rels]) + ' _%s' % e + ' )' * depth
    data.add((x, y))
  return list(data)

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

# Augmentation Routines
# Don't use any external information besides what's in the datsaet
# and entity alignments
# Assume that there's a single entity in each example.
def get_templates(dataset):
  def create_template(ex):
    x, y = ex
    x_new = re.sub('ent:[0-9]{2}', '%s', x)
    y_new = re.sub('_ent:[0-9]{2}', '%s', y)
    return (x_new, y_new)
  templates = [create_template(ex) for ex in dataset]
  return templates

def augment_nesting(dataset):
  # Augment by ensting one thing within another
  def combine(template, ex):
    x_t, y_t = template
    x_e, y_e = ex
    x_new = x_t % x_e
    y_new = y_t % y_e
    return (x_new, y_new)
  templates = get_templates(dataset)
  new_examples = [combine(t, ex) for t in templates for ex in dataset]
  return new_examples

def augment_entities(dataset):
  # Augment by swapping in new entity names
  def replace(template, ent):
    x_t, y_t = template
    x_new = x_t % ent
    y_new = y_t % ('_' + ent)
    return (x_new, y_new)
  entities = sorted(list(set(re.search('ent:[0-9]{2}', x).group(0)
                             for x, y in dataset)))
  templates = get_templates(dataset)
  new_examples = [replace(t, e) for t in templates for e in entities]
  return new_examples

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
  write_data('augNested/augNested_test500.tsv', nested_test)

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

  # Augmented datasets
  def write_augmented(dirbase, dataset, aug_method, nums=None):
    if aug_method == 'none':
      new_data = []
    elif aug_method == 'nesting':
      new_data = augment_nesting(dataset)
    elif aug_method == 'entities':
      new_data = augment_entities(dataset)
    elif aug_method == 'both':
      new_data = augment_entities(augment_nesting(dataset))
    else:
      raise ValueError('Unrecognzied augmentation method "%s"' % aug_method)
    print >> sys.stderr, 'Augmentation mode "%s": %d new examples' % (
        aug_method, len(new_data))
    random.shuffle(new_data)
    if nums:
      for n in nums:
        cur_data = new_data[:n]
        filename = '%s_aug%s%03d.tsv' % (dirbase, aug_method.title(), n)
        write_data(filename, dataset + cur_data)
    else:
      filename = '%s_aug%s.tsv' % (dirbase, aug_method.title())
      write_data(filename, dataset + new_data)

  nested_train100 = nested_train[:100]
  aug_nums = (25, 50, 75, 100, 150, 200, 250, 300, 400, 500)
  dirbase = 'augNested/train_nested100'
  write_augmented(dirbase, nested_train100, 'none')
  write_augmented(dirbase, nested_train100, 'nesting', aug_nums)
  write_augmented(dirbase, nested_train100, 'entities', aug_nums)
  write_augmented(dirbase, nested_train100, 'both', aug_nums)
  depth4 = sample_nested(depth=4, num=1000)
  for i in aug_nums:
    write_data('augNested/train_nested%d.tsv' % (100 + i), nested_train[:100+i])
    write_data('augNested/train_nested100_deeper%d.tsv' % i, nested_train100 + depth4[:i])

if __name__ == '__main__':
  main()
