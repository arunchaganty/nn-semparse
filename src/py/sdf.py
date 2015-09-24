"""Parses the simple data format of SEMPRE."""
import collections


Record = collections.namedtuple(
    'Record', 
    ['utterance', 'canonical_utterance', 'compatibility', 'formula', 'prob'])


def read(filename):
  utterance_list = list()
  utterance_to_record = collections.defaultdict(list)
  with open(filename) as f:
    for line in f:
      if not line.startswith('Pred'): continue
      toks = line.strip().split('\t')[1:]
      toks[2] = float(toks[2])  # compatibility
      toks[4] = float(toks[4])  # prob
      record = Record(*toks)
      if record.utterance not in utterance_list:
        utterance_list.append(record.utterance)
      utterance_to_record[record.utterance].append(record)
  return [utterance_to_record[x] for x in utterance_list]

def get_best_correct(records):
  """Get single highest-weight correct parse."""
  correct_records = [r for r in records if r.compatibility]
  if not correct_records: return None
  return max(correct_records, key=lambda r: r.prob)
