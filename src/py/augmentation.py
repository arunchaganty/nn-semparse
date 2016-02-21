"""Module that handles all augmentation."""
import collections
import random
import sys

import atisaugment
import atislexicon
import geoaugment
import regexaugment

class Augmenter(object):
  def __init__(self, dataset, unk_cutoff=1):
    self.dataset = dataset
    self.lexicon = self.get_lexicon()

    # Set up UNK processing
    self.unk_cutoff = unk_cutoff
    self.next_unk = 1
    sentences = [x for x, y in self.dataset]
    counts = collections.Counter()
    for s in sentences:
      counts.update(s.split(' '))
    self.vocab = set(w for w in counts if counts[w] > unk_cutoff)

    # Set up templates and replacements for augmentation
    self.setup_templates_and_replacements()

    # Do other setup
    self.setup()

  def get_lexicon(self):
    """By default, no lexicon."""
    return None

  def setup_templates_and_replacements(self):
    # TODO(robinjia): use the lexicon
    templates = set()
    replacements = set()
    for x, y in self.dataset:
      x_toks = x.split(' ')
      y_toks = y.split(' ')
      x_holes = []
      y_holes = []
      cur_num_reps = 0

      if self.lexicon:
        lex_items = self.lexicon.map_over_sentence(x_toks, return_entries=True)
      else:
        lex_items = [((i, i+1), x_toks[i]) for i in range(len(x_toks))]
      for (i, j), ent in lex_items:
        # Make sure this entity only occurs once in x and y
        x_span = ' '.join(x_toks[i:j])
        if x_toks.count(x_span) != 1: continue
        if y_toks.count(ent) != 1: continue

        # Add the replacement rule
        replacements.add((x_span, ent))

        # Update the template
        x_holes.append((i, i+1))
        y_holes.append(y_toks.index(ent))
        cur_num_reps += 1

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
      templates.add((x_t, y_t, cur_num_reps))

    self.templates = list(templates)
    self.replacements = list(replacements)

    for t in templates:
      print t
    for r in replacements:
      print r
    print 'Found %d simple templates' % len(self.templates)
    print 'Found %d simple replacements' % len(self.replacements)

  def setup(self):
    pass

  def _mask_unk(self, x):
    """Mask UNK words, replace with an UNK token."""
    toks = x.split(' ')
    new_toks = []
    unk_dict = {}
    for t in toks:
      if t in self.vocab:
        new_toks.append(t)
      else:
        if t not in unk_dict:
          unk_dict[t] = 'unk:%06d:%s' % (self.next_unk, t)
          self.next_unk += 1
        new_toks.append(unk_dict[t])
    x_new = ' '.join(new_toks)
    return x_new

  def sample_sentence(self, mask_unk=False):
    x_t, y_t, n = random.sample(self.templates, 1)[0]
    cur_reps = random.sample(self.replacements, n)
    x_reps = dict(('w%d' % i, cur_reps[i][0]) for i in range(n))
    y_reps = dict(('w%d' % i, cur_reps[i][1]) for i in range(n))
    x_new = x_t % x_reps
    y_new = y_t % y_reps
    if mask_unk:
      x_new = self._mask_unk(x_new)
    return (x_new, y_new)

  def augment_single(self, num, mask_unk=False):
    aug_data = set()
    while len(aug_data) < num:
      x, y = self.sample_sentence(mask_unk=mask_unk)
      aug_data.add((x, y))
    return list(aug_data)

  def augment_double(self, num, mask_unk=False):
    aug_data = set()
    while len(aug_data) < num:
      x1, y1 = self.sample_sentence(mask_unk=mask_unk)
      x2, y2 = self.sample_sentence(mask_unk=mask_unk)
      x_new = '%s <sep> %s' % (x1, x2)
      y_new = '%s <sep> %s' % (y1, y2)
      aug_data.add((x_new, y_new))
    return list(aug_data)

  def augment_concat(self, num, mask_unk=False):
    """Simple augmentation by concatenating two input examples."""
    aug_data = []
    aug_set = set()
    while len(aug_data) < num:
      (x1, y1), (x2, y2) = random.sample(self.dataset, 2)
      if mask_unk:
        x1, x2 = self._mask_unk(x1), self._mask_unk(x2)
      x_new = '%s <sep> %s' % (x1, x2)
      y_new = '%s <sep> %s' % (y1, y2)
      if (x_new, y_new) not in aug_set:
        aug_data.append((x_new, y_new))
        aug_set.add((x_new, y_new))
    return aug_data

  def augment(self, aug_type, num):
    if aug_type == 'concat':
      return self.augment_concat(num)
    elif aug_type == 'concat-mask':
      return self.augment_concat(num, mask_unk=True)
    elif aug_type == 'single':
      return self.augment_single(num)
    elif aug_type == 'single-mask':
      return self.augment_single(num, mask_unk=True)
    elif aug_type == 'double':
      return self.augment_double(num)
    elif aug_type == 'double-mask':
      return self.augment_double(num, mask_unk=True)
    else:
      return self.augment_special(aug_type, num)

  def augment_special(self, aug_type, num):
    """Override this to define custom domain-specific augmentation modes."""
    raise ValueError('Unrecognized augmentation type "%s"' % aug_type)

class GeoAugmenter(Augmenter):
  """Augmentation for geoquery.

  Recognized augmentatino types: ['pcfg'].
  """
  def augment_special(self, aug_type, num):
    if aug_type == 'pcfg':
      return geoaugment.sample_pcfg(self.dataset, num)
    else:
      raise ValueError('Unrecognized augmentation type "%s"' % aug_type)

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

  def augment_special(self, aug_type, num):
    if aug_type == 'str':
      return self.augment_str(num)
    elif aug_type == 'int':
      return self.augment_int(num)
    elif aug_type == 'conj':
      return self.augment_conj(num)
    else:
      raise ValueError('Unrecognized augmentation type "%s"' % aug_type)

class AtisAugmenter(Augmenter):
  def setup(self):
    self.atis_templates, self.atis_replacements = atisaugment.get_templates_and_replacements(self.dataset)

  def get_lexicon(self):
    return atislexicon.get_lexicon()

  def augment_atis_single(self, num):
    return atisaugment.augment_single(self.atis_templates, self.atis_replacements, num)

  def augment_atis_double(self, num):
    return atisaugment.augment_double(self.atis_templates, self.atis_replacements, num)

  def augment_special(self, aug_type, num):
    if aug_type == 'atis-single':
      return self.augment_atis_single(num)
    elif aug_type == 'atis-double':
      return self.augment_atis_double(num)
    else:
      raise ValueError('Unrecognized augmentation type "%s"' % aug_type)


def new(domain, dataset):
  if domain == 'geoquery':
    return GeoAugmenter(dataset)
  elif domain == 'regex':
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
  for ex in aug_data:
    print '\t'.join(ex)

if __name__ == '__main__':
  main()
