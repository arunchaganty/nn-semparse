"""Grammar for geoquery."""
import numpy
import sys

from grammar import Grammar, Rule

class GeoGrammar(Grammar):
  def create_generator(self):
    """Variables are A, B, C, etc."""
    cur = 'A'
    while True:
      yield cur
      cur = chr(ord(cur) + 1)

def city_rule(name):
  if ' ' in name:
    name = "' %s '" % name
  return Rule('$City', [], name,
              lambda v: lambda var: ('_const ( %s , _cityid ( ' % var) + name + ' , _ ) )',
              weight=0.05)

def river_rule(name):
  orig_name = name
  if name.endswith(' river'):
    name = name[:-6]
  if ' ' in name:
    name = "' %s '" % name
  return Rule('$River', [], orig_name,
              lambda v: lambda var: ('_const ( %s , _riverid ( ' % var) + name + ' ) )')

RULES = [
    # Root rules 
    Rule('$ROOT', ['$Answer'], 'what %s ?', lambda v, a: lambda : '_answer ( %(v1)s , ( %(c1)s ) )' % {'v1': v.get(1), 'c1': a.sem_fn(v.get(1))}),
    #Rule('$ROOT', ['$Answer'], 'what is the %s ?', lambda v, a: lambda : '_answer ( %(v1)s , ( %(c1)s ) )' % {'v1': v.get(1), 'c1': a.sem_fn(v.get(1))}),
    #Rule('$ROOT', ['$Answer'], 'what are the %s ?', lambda v, a: lambda : '_answer ( %(v1)s , ( %(c1)s ) )' % {'v1': v.get(1), 'c1': a.sem_fn(v.get(1))}),

    # Things that could be answers
    Rule('$Answer', ['$State'], '%s', lambda v, a: lambda var: a.sem_fn(var)),
    Rule('$Answer', ['$Landmark'], '%s', lambda v, a: lambda var: a.sem_fn(var)),
    Rule('$Answer', ['$State'], 'population of %s',
         lambda v, a: lambda var: '_population ( %(v1)s , %(var)s ) , %(c1)s' % {'var': var, 'v1': v.get(1), 'c1': a.sem_fn(v.get(1))}),
    Rule('$Answer', ['$State'], 'highest points in %s',
         lambda v, a: lambda var: '_highest ( %(var)s , ( _place ( %(var)s ) , _loc ( %(var)s , %(v1)s ) , %(c1)s' % {'var': var, 'v1': v.get(1), 'c1': a.sem_fn(v.get(1))}),
    Rule('$Answer', ['$State'], 'lowest points in %s',
         lambda v, a: lambda var: '_lowest ( %(var)s , ( _place ( %(var)s ) , _loc ( %(var)s , %(v1)s ) , %(c1)s' % {'var': var, 'v1': v.get(1), 'c1': a.sem_fn(v.get(1))}),
    Rule('$Answer', ['$River'], 'length of %s', lambda v, a: lambda var: '_len ( %(var)s , %(v1)s ) , %(c1)s' % {'var': var, 'v1': v.get(1), 'c1': a.sem_fn(v.get(1))}),


    # States
    Rule('$State', [], 'florida', lambda v: lambda var: '_const ( %s , _stateid ( florida ) )' % var),
    Rule('$State', [], 'maine', lambda v: lambda var: '_const ( %s , _stateid ( maine ) )' % var),
    Rule('$State', [], 'mississippi', lambda v: lambda var: '_const ( %s , _stateid ( mississippi ) )' % var),
    Rule('$State', [], 'texas', lambda v: lambda var: '_const ( %s , _stateid ( texas ) )' % var),
    Rule('$State', [], 'oregon', lambda v: lambda var: '_const ( %s , _stateid ( oregon ) )' % var),
    Rule('$State', [], 'illinois', lambda v: lambda var: '_const ( %s , _stateid ( illinois ) )' % var),
    Rule('$State', ['$State'], 'states border %s',
         lambda v, a: lambda var: '_state ( %(var)s ) , _next_to ( %(var)s , %(v1)s ) , %(c1)s' % {'var': var, 'v1': v.get(1), 'c1': a.sem_fn(v.get(1))}),
    Rule('$State', ['$State'], 'states that border %s',
         lambda v, a: lambda var: '_state ( %(var)s ) , _next_to ( %(var)s , %(v1)s ) , %(c1)s' % {'var': var, 'v1': v.get(1), 'c1': a.sem_fn(v.get(1))}),
    Rule('$State', ['$Landmark'], 'state that has %s',
         lambda v, a: lambda var: '_state ( %(var)s ) , _loc ( %(v1)s , %(var)s ) , %(c1)s' % {'var': var, 'v1': v.get(1), 'c1': a.sem_fn(v.get(1))}),
    Rule('$State', ['$River'], 'states through which the %s run', 
         lambda v, a: lambda var: '_state ( %(var)s ) , %(c1)s , _traverse ( %(v1)s , %(var)s )' % {'var': var, 'v1': v.get(1), 'c1': a.sem_fn(v.get(1))}),
    Rule('$State', [], 'the largest state', lambda v: lambda var: '_largest ( %(var)s , _state ( %(var)s ) )' % {'var': var}),
    Rule('$State', [], 'the smallest state', lambda v: lambda var: '_smallest ( %(var)s , _state ( %(var)s ) )' % {'var': var}),
    Rule('$State', ['$State'], 'not %s', lambda v, a: lambda var: '\\+ ( %(c1)s )' % {'c1': a.sem_fn(var)}),

    # Landmarks
    Rule('$Landmark', ['$City'], '%s', lambda v, a: lambda var: a.sem_fn(var)),
    Rule('$Landmark', ['$River'], '%s', lambda v, a: lambda var: a.sem_fn(var)),
    

    # Complex city rules
    Rule('$City', ['$State'], 'capital of %s',
         lambda v, a: lambda var: '_capital ( %(var)s ) , _loc ( %(var)s , %(v1)s ) , %(c1)s' % {'var': var, 'v1': v.get(1), 'c1': a.sem_fn(v.get(1))}),
    Rule('$City', ['$State'], 'major cities in %s',
         lambda v, a: lambda var: '_major ( %(var)s ) , _city ( %(var)s ) ,  _loc ( %(var)s , %(v1)s ) , %(c1)s' % {'var': var, 'v1': v.get(1), 'c1': a.sem_fn(v.get(1))}),
    Rule('$City', ['$City'], 'not %s', lambda v, a: lambda var: '\\+ ( %(c1)s )' % {'c1': a.sem_fn(var)}),

    # Complex river rules
    Rule('$River', ['$State'], 'rivers in %s',
         lambda v, a: lambda var: '_river ( %(var)s ) , _loc ( %(var)s , %(v1)s ) , %(c1)s )' % {'var': var, 'v1': v.get(1), 'c1': a.sem_fn(v.get(1))}),
    Rule('$River', ['$State'], 'longest river in %s',
         lambda v, a: lambda var: '_longest ( %(var)s , ( _river ( %(var)s ) , _loc ( %(var)s , %(v1)s ) , %(c1)s ) )' % {'var': var, 'v1': v.get(1), 'c1': a.sem_fn(v.get(1))}),
]


CITIES = [
    'albany', 'atlanta', 'austin', 'boston', 'boulder', 'columbus', 'dallas', 
    'denver', 'des moines', 'dover', 'flint', 'fort wayne', 'houston',
    'indianapolis', 'kalamazoo', 'montgomery', 'new orleans', 'pittsburgh',
    'portland', 'rochester', 'sacramento', 'salem', 'san diego', 'san francisco',
    'san jose', 'scotts valley', 'seattle', 'spokane'
]
RULES += [city_rule(c) for c in CITIES]

RIVERS = ['colorado river', 'mississippi river', 'missouri river',
          'potomac river', 'red river', 'rio grande']
RULES += [river_rule(r) for r in RIVERS]

def main(num_samples):
  grammar = GeoGrammar(RULES)
  for i in range(num_samples):
    deriv = grammar.sample()
    print '%s\t%s' % (deriv.text, deriv.sem)

if __name__ == '__main__':
  numpy.random.seed(1)  # For determinism
  if len(sys.argv) == 1:
    print >> sys.stderr, 'Usage: %s [num samples]' % sys.argv[0]
    sys.exit(1)
  num_samples = int(sys.argv[1])
  main(num_samples)
