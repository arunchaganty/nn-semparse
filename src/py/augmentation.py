"""Module that handles all augmentation."""
import random
import sys

import regexaugment

class Augmenter(object):
  def __init__(self, dataset):
    self.dataset = dataset
    self.setup()

  def setup(self):
    pass

  def augment_concat(self, num):
    """Simple augmentation by concatenating two input examples."""
    aug_data = set()
    while len(aug_data) < num:
      (x1, y1), (x2, y2) = random.sample(self.dataset, 2)
      x_new = '%s <sep> %s' % (x1, x2)
      y_new = '%s <sep> %s' % (y1, y2)
      aug_data.add((x_new, y_new))
    return list(aug_data)

  def augment(self, aug_type, num):
    raise NotImplementedError

class RegexAugmenter(Augmenter):
  """Augmentation for regex

  Recognized augmentation types: ['str', 'int', 'conj'].
  """
  def setup(self):
    self.str_templates = regexaugment.get_str_templates(self.dataset)
    self.int_templates = regexaugment.get_int_templates(self.dataset)

  def augment_str(self, num):
    return regexaugment.augment_str(self.dataset, self.str_templates, num)

  def augment_int(self, num):
    return regexaugment.augment_int(self.dataset, self.int_templates, num)

  def augment_conj(self, num):
    return regexaugment.augment_conj(self.dataset, self.str_templates,
                                     self.int_templates, num)

  def augment(self, aug_type, num):
    if aug_type == 'concat':
      return self.augment_concat(num)
    elif aug_type == 'str':
      return self.augment_str(num)
    elif aug_type == 'int':
      return self.augment_int(num)
    elif aug_type == 'conj':
      return self.augment_conj(num)
    else:
      raise ValueError('Unrecognized augmentation type "%s"' % aug_type)

class AtisAugmenter(Augmenter):
  def augment(self, aug_type, num):
    if aug_type == 'concat':
      return self.augment_concat(num)
    else:
      raise ValueError('Unrecognized augmentation type "%s"' % aug_type)

def new(domain, dataset):
  if domain == 'regex':
    return RegexAugmenter(dataset)
  elif domain == 'atis':
    return AtisAugmenter(dataset)
  raise ValueError('Unrecognzied domain "%s"' % domain)

def main():
  """Print augmented data to stdout."""
  if len(sys.argv) < 5:
    print >> sys.stderr, 'Usage: %s [file] [domain] [aug-type] [num]' % sys.argv[0]
    sys.exit(1)
  fname, domain, aug_type, num = sys.argv[1:5]
  num = int(num)
  with open(fname) as f:
    data = [x.strip().split('\t') for x in f]
  augmenter = new(domain, data)
  aug_data = augmenter.augment(aug_type, num)
  for ex in data + aug_data:
    print '\t'.join(ex)

if __name__ == '__main__':
  main()
