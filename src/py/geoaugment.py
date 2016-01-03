"""Augment geoquery data.

Rule: replace states with state-typed entities.
"""
import collections
import glob
import os
import random
import re
import sys

IN_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'data/geo880/processed-lesscopy')
OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'data/geo880/processed-augmented')

def read_examples(filename):
  with open(filename) as f:
    data = [tuple(line.strip().split('\t')) for line in f]
  print 'Read %d examples' % len(data)
  return data

def extract_type(y):
  if '_city ( A )' in y or '_capital ( A )' in y:
    return 'city'
  elif '_river ( A )' in y:
    return 'river'
  elif '_state ( A )' in y:
    return 'state'
  return None

def prune_utterance(x):
  patterns = [
      '\\?$',
      '^in',
      '^what is',
      '^what are',
      '^what',
      '^which',
      '^name',
      '^give me',
      '^can you tell me',
      '^show',
  ]
  old_x = x
  for pattern in patterns:
    x = re.sub(pattern, '', x).strip()
  if x == old_x or x == old_x[:-2]:
    print >> sys.stderr, 'Could not prune: "%s"' % old_x
    return None
  return x

def prune_lf(y):
  m = re.match('_answer \\( A , (.*) \\)', y)
  if not m:
    print >> sys.stderr, 'Bad logical form "%s"' % y
    return None
  lf = m.group(1)
  m2 = re.match('\\( (.*) \\)', lf)
  if m2:
    lf = m2.group(1)
  return lf

def get_replacers(in_data):
  """Get map of type -> list of (x, y) that can be inserted into other examples.

  type is either 'city', 'river', or 'state'.
  """
  ret = collections.defaultdict(set)
  for x, y in in_data:
    cur_type = extract_type(y)
    if not cur_type: continue
    utterance = prune_utterance(x)
    if utterance is None: continue
    lf = prune_lf(y)
    if not lf: continue
    # print '%s: "%s", "%s"' % (cur_type, utterance, lf)
    ret[cur_type].add((utterance, lf))
  for t in ret:
    print >> sys.stderr, 'Replacements of type %s: %d entries' % (t, len(ret[t]))
  return ret

def clean_name(name):
  return name.split(',')[0].replace("'", '').strip()

def make_template(x, y):
  m = re.search("_const \\( ([A-Z]) , _([a-z]*)id \\( ([^)]*) \\) \\)", y)
  if not m: return None
  expr = m.group(0)
  varname = m.group(1)
  cur_type = m.group(2)
  name = m.group(3)
  if cur_type not in ('city', 'river', 'state'): return None
  if cur_type == 'city' and '_' not in name: return None
  name = clean_name(name)
  if cur_type == 'river' and name + ' river' in x:
    x_template = x.replace(name + ' river', '%s')
  else:
    x_template = x.replace(name, '%s')
  if x_template == x:
    print >> sys.stderr, 'Could not locate "%s" in "%s"' % (name, x)
  y_template = y.replace(m.group(0), '%s')
  return (cur_type, x_template, y_template, varname)

def get_templates(in_data):
  """Get map of type -> (x, y, var) triples that can be used as templates.

  type is either 'city', 'river', or 'state'.

  The returned x and y are format strings with a single "%s".
  var is the variale that should be used for the replacement.
  """
  ret = collections.defaultdict(set)
  for x, y in in_data:
    template = make_template(x, y)
    if template:
      cur_type, x_template, y_template, varname = template
      ret[cur_type].add((x_template, y_template, varname))
  for t in ret:
    print >> sys.stderr, 'Templates of type %s: %d entries' % (t, len(ret[t]))
  return ret

def find_next_var(y, known_var):
  toks = y.split()
  var = chr(ord(known_var) + 1)
  while var in toks:
    var = chr(ord(var) + 1)
  return var

def plug_in_vars(y_r, main_var, next_var):
  orig_toks = y_r.split()
  new_toks = y_r.split()
  var = 'A'
  while var in orig_toks:
    if var == 'A':
      replacement = main_var
    else:
      replacement = next_var
      next_var = chr(ord(next_var) + 1)
    for i in range(len(orig_toks)):
      if orig_toks[i] == var:
        new_toks[i] = replacement
    var = chr(ord(var) + 1)
  return ' '.join(new_toks)

def merge(template, replacement):
  x_t, y_t, varname = template
  x_r, y_r = replacement
  next_var = find_next_var(y_t, varname)
  y_r = plug_in_vars(y_r, varname, next_var)
  x = x_t % x_r
  y = y_t % y_r
  return (x, y)

def augment_data(in_data):
  replacers = get_replacers(in_data)
  templates = get_templates(in_data)
  augmented_data = []
  for cur_type in replacers:
    cur_replacers = replacers[cur_type]
    cur_templates = templates[cur_type]
    for r in cur_replacers:
      for t in cur_templates:
        augmented_data.append(merge(t, r))
  return augmented_data

def write_data(basename, data):
  out_path = os.path.join(OUT_DIR, basename)
  with open(out_path, 'w') as f:
    for x, y in data:
      print >>f, '%s\t%s' % (x, y)

def write_sample(in_data, augmented_data, k):
  n = len(in_data)
  if k > len(augmented_data):
    print >> sys.stderr, 'Trying to sample %d from dataset of size %d' % (
        k, len(augmented_data))
    return
  sampled_data = random.sample(augmented_data, k)
  if k % 1000 == 0:
    out_num = '%dk' % (k / 1000)
  else:
    out_num = '%d' % k
  write_data('geo880_train%d_augment%s.tsv' % (n, out_num), in_data + sampled_data)

def extract_entity(x, y):
  # WARNING: copying code from make_template(x, y)
  m = re.search("_const \\( ([A-Z]) , _([a-z]*)id \\( ([^)]*) \\) \\)", y)
  if not m: return None
  expr = m.group(0)
  varname = m.group(1)
  cur_type = m.group(2)
  name = m.group(3)
  if cur_type not in ('city', 'river', 'state'): return None
  if cur_type == 'city' and '_' not in name: return None
  e_y = re.sub('[A-Z]', 'A', expr)
  e_x = clean_name(name)
  if cur_type == 'river' and name + ' river' in x:
    e_x = e_x + ' river'
  return (cur_type, e_x, e_y)

def get_entities(in_data):
  ret = collections.defaultdict(set)
  for x, y in in_data:
    ent = extract_entity(x, y)
    if ent:
      cur_type, e_x, e_y = ent
      ret[cur_type].add((e_x, e_y))
  for t in ret:
    print >> sys.stderr, 'Entities of type %s: %d entries' % (t, len(ret[t]))
  return ret

def sample_pcfg(in_data, n):
  replacers = get_replacers(in_data)
  templates = get_templates(in_data)
  entities = get_entities(in_data)
  augmented_data = set()
  while len(augmented_data) < n:
    cur_type = random.sample(list(replacers), 1)[0]
    r = random.sample(replacers[cur_type], 1)[0]
    t = random.sample(templates[cur_type], 1)[0]
    r_template = make_template(r[0], r[1])
    if r_template:
      inner_type, r_x, r_y, varname = r_template
      e = random.sample(entities[inner_type], 1)[0]
      r_new = merge((r_x, r_y, varname), e)
      #print '%s -> %s' % (r, r_new)
      r = r_new
    new_ex = merge(t, r)
    augmented_data.add(new_ex)
  return list(augmented_data)

def write_sublist(in_data, augmented_data, k):
  n = len(in_data)
  if k % 1000 == 0:
    out_num = '%dk' % (k / 1000)
  else:
    out_num = '%d' % k
  new_data = in_data + augmented_data[:k]
  write_data('geo880_train%d_augpcfg%s.tsv' % (n, out_num), new_data)


def process(filename):
  random.seed(1)
  print >> sys.stderr, 'Processing %s' % filename
  basename = os.path.basename(filename)
  in_data = read_examples(filename)
  n = len(in_data)
  augmented_data = augment_data(in_data)

  # Write everything
  write_data('geo880_train%d_augmentAll.tsv' % n, in_data + augmented_data)

  # Use a weird order to make sure randomness is same as previous
  write_sample(in_data, augmented_data, 1000)
  write_sample(in_data, augmented_data, 5000)
  write_sample(in_data, augmented_data, 2000)
  write_sample(in_data, augmented_data, 4000)
  write_sample(in_data, augmented_data, 500)
  write_sample(in_data, augmented_data, 3000)
  write_sample(in_data, augmented_data, 250)
  write_sample(in_data, augmented_data, 750)
  write_sample(in_data, augmented_data, 1500)

  # Use the new augmentation strategy, augpcfg
  augpcfg_data = sample_pcfg(in_data, 2000)
  write_sublist(in_data, augpcfg_data, 250)
  write_sublist(in_data, augpcfg_data, 500)
  write_sublist(in_data, augpcfg_data, 750)
  write_sublist(in_data, augpcfg_data, 1000)
  write_sublist(in_data, augpcfg_data, 1500)
  write_sublist(in_data, augpcfg_data, 2000)

def main():
  for n in (500, 100, 200, 300, 400, 600):
    filename = os.path.join(IN_DIR, 'geo880_train%d.tsv' % n)
    process(filename)

if __name__ == '__main__':
  main()
