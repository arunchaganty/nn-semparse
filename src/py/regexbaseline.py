"""A simple template-matching regex baseline.

Just to see how well this does..."""
import os
import sys

def read(filename):
  with open(filename) as f:
    return [tuple(line.strip().split('\t')) for line in f]

def make_pattern(x):
  if x.endswith('.'): x = x[:-1]
  toks = x.split()
  if toks[-1] == '.': toks = toks[:-1]
  new_toks = []
  strs = []
  in_quote = None
  cur_str_toks = []
  for t in toks:
    if t == "'" or t == '"':
      if in_quote == t: 
        in_quote = None
        new_toks.append('%s')
        strs.append(' '.join(cur_str_toks))
        cur_str_toks = []
      elif not in_quote: in_quote = t
      else: cur_str_toks.append(t)
    else:
      if in_quote: cur_str_toks.append(t)
      else: 
        if len(t) == 1 and ord(t) >= ord('0') and ord(t) <= ord('9'):
          strs.append(t)
        else:
          new_toks.append(t)
  return (' '.join(new_toks), strs)

def extract_templates(dataset):
  templates = {}
  for x, y in dataset:
    pattern, strs = make_pattern(x)
    y_toks = y.split()
    if any(s not in y_toks for s in strs): continue
    new_y_toks = []
    for t in y_toks:
      for i, s in enumerate(strs):
        if s == t:
          new_y_toks.append('%(' + str(i) + ')s')
          break
      else:
        new_y_toks.append(t)
    new_y = ' '.join(new_y_toks)
    if pattern in templates and templates[pattern] != new_y:
      print >> sys.stderr, 'Collision for "%s": ' % pattern
      print >> sys.stderr, '  old: %s' % templates[pattern]
      print >> sys.stderr, '  new: %s' % new_y
    templates[pattern] = new_y
  return templates

def list_to_dict(l):
  return dict((str(i), val) for i, val in enumerate(l))

def main(train_file, dev_file):
  train_data = read(train_file)
  dev_data = read(dev_file)
  train_templates = extract_templates(train_data)

  num_correct = 0
  num_pred = 0
  for x, y in dev_data:
    pattern, strs = make_pattern(x)
    if pattern in train_templates:
      print '    Found: %s' % pattern
      y_pred = train_templates[pattern] % list_to_dict(strs)
      num_pred += 1
      if y_pred == y:
        num_correct += 1
    else:
      print 'Not found: %s' % pattern

  # Print stats
  print 'Precision: %d/%d = %g' % (
      num_correct, num_pred, float(num_correct) / num_pred)
  print 'Recall: %d/%d = %g' % (
      num_correct, len(dev_data), float(num_correct) / len(dev_data))


if __name__ == '__main__':
  if len(sys.argv) == 1:
    print >> sys.stderr, 'Usage: %s regex_train.tsv regex_dev.tsv' % sys.argv[0]
    sys.exit(1)
  main(*sys.argv[1:])
