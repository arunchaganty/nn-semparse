"""Module that handles all augmentation."""
import regexaugment

class Augmenter(object):
  def __init__(self, dataset):
    self.dataset = dataset
    self.setup()

  def setup(self):
    pass

  def augment(self, aug_type, num):
    raise NotImplementedError

class RegexAugmenter(Augmenter):
  """Augmentation for regex

  Recognized augmentation types: ['str', 'int'].
  """
  def setup(self):
    self.str_templates, self.int_templates = regexaugment.get_templates(self.dataset)

  def augment_str(num):
    return regexaugment.augment_str(self.dataset, self.str_templates, num)

  def augment_int(num):
    return regexaugment.augment_int(self.dataset, self.int_templates, num)

  def augment(self, aug_type, num):
    if aug_type == 'str':
      return self.augment_str(num)
    elif aug_type == 'int':
      return self.augment_int(num)
    else:
      raise ValueError('Unrecognized augmentation type "%s"' % aug_type)

def new(domain, dataset):
  if domain == 'regex':
    return RegexAugmenter(dataset)
  raise ValueError('Unrecognzied domain "%s"' % domain)
