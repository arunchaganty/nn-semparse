"""A Grammar (a la SEMPRE)."""
import collections
import numpy
import re

class Grammar(object):
  ROOT = '$ROOT'

  def __init__(self, rules):
    self.rules = rules
    all_lhs = set(r.lhs for r in rules)

    self.cat_to_rules = collections.defaultdict(list)
    for r in rules:
      self.cat_to_rules[r.lhs].append(r)

    # Do some error checking
    if self.ROOT not in self.cat_to_rules:
      raise ValueError('No rules added for required category "%s"' % self.ROOT)
    all_rhs = set(c for r in rules for c in r.rhs)
    for rhs in all_rhs:
      if rhs not in self.cat_to_rules:
        raise ValueError('Category "%s" used as a RHS, '
                         'but no rules have it as LHS' % rhs)

  def sample(self):
    """Generate a sample from this grammar."""
    def recurse(cat, depth):
      cur_rules = self.cat_to_rules[cat]
      total_weight = float(sum(r.weight for r in cur_rules))
      rule = numpy.random.choice(
          cur_rules, p=[r.weight / total_weight for r in cur_rules])
      child_values = [recurse(c, depth+1) for c in rule.rhs]
      return rule.apply_to(child_values, )
    root_deriv = recurse(self.ROOT, 0)
    root_deriv.sem = VariableGenerator.normalize(root_deriv.sem_fn(),
                                                    self.create_generator())
    return root_deriv

  def create_generator(self):
    """Should be implemented by concrete sublcasses."""
    raise NotImplementedError


class Derivation(object):
  def __init__(self, text, sem_fn):
    """A derivation

    Note that partial derivations have a sem_fn with holes to fill in (variable names).
    Full derivations also have sem, which is a full logical form.
    """
    self.text = text
    self.sem_fn = sem_fn
    self.sem = None

  def __str__(self):
    return str((self.text, self.sem))

class Rule(object):
  def __init__(self, lhs, rhs, text_pattern, sem_fn, weight=1.0):
    """A grammar rule

    Args:
      self.lhs: left hand side category (e.g. '$ROOT')
      self.rhs: list of right hand side categories
      self.text_pattern: pattern to generate text (e.g. '%s of %s')
      self.sem_fn: function that generates semantics.
        Takes as first argument a VariableGenerator,
        then subsequent arguments are child derivations.
        e.g. lambda v, a, b '(and %s %s %s)' % (v.get(0), a.sem, b.sem)
    """
    self.lhs = lhs
    self.rhs = rhs
    self.text_pattern = text_pattern
    self.sem_fn = sem_fn
    self.weight = weight
    self.vg = VariableGenerator.new()

  def apply_text(self, child_derivs):
    texts = tuple(d.text for d in child_derivs)
    return self.text_pattern % texts

  def apply_sem(self, child_derivs):
    """Generate new logical form given child logical forms."""
    return self.sem_fn(self.vg, *child_derivs)

  def apply_to(self, child_derivs):
    text = self.apply_text(child_derivs)
    sem = self.apply_sem(child_derivs)
    return Derivation(text, sem)

class VariableGenerator(object):
  next_uid = 0

  def __init__(self, uid):
    """WARNING: should not be accessed directly; use VariableGenerator.new()."""
    self.uid = uid

  @classmethod
  def new(cls):
    vg = cls(cls.next_uid)
    cls.next_uid += 1
    return vg

  def get(self, i):
    return 'var:%d-%d' % (self.uid, i)

  @classmethod
  def normalize(cls, s, gen):
    """Convert a string with variable names taken from the given generator.
    
    Assumes that s is space-delimited tokens.
    """
    toks = s.split(' ')
    new_toks = []
    var_map = {}
    for t in toks:
      m = re.match('var:[0-9]+-[0-9]+', t)
      if m:
        if t not in var_map:
          var_map[t] = next(gen)
        new_toks.append(var_map[t])
      else:
        new_toks.append(t)
    return ' '.join(new_toks)
