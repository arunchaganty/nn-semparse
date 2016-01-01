#!/usr/bin/env python
import glob
import os
import re
import subprocess
import sys
import time

if len(sys.argv) < 4:
  print >> sys.stderr, 'Usage: %s [condition] [hidden layer size] [word embedding size]' % sys.argv[0]
  sys.exit(1)

condition = sys.argv[1]
d = int(sys.argv[2])
i = int(sys.argv[3])
device = 'cpu'
num_cpu = 1

def launch(filename):
  # Filenames are like train_nested150.tsv
  basename = os.path.basename(filename)
  cl_filename = os.path.join('artificial', condition, basename)
  description = basename[6:-4]
  vals = {'d': d, 'i': i, 'device': device, 'num_cpu': num_cpu, 'filename': cl_filename, 'description': description, 'condition': condition}
  #print ' '.join([
  subprocess.call([
      'cl', 'run', ':src', ':artificial',
      'OMP_NUM_THREADS=%(num_cpu)d THEANO_FLAGS=blas.ldflags=-lopenblas,device=%(device)s,floatX=float32 python src/py/main.py -d %(d)d -i %(i)d -o %(i)d -p attention -u 0 -t 30 -c lstm -m attention --stats-file stats.json --train-data %(filename)s --dev-data artificial/%(condition)s/%(condition)s_test500.tsv --save-file params' % vals,
      '--request-queue', 'jag', '--request-cpus', str(num_cpu), '-n', 'artificial-%s' % condition, '-d', description])
  time.sleep(1)

filenames = sorted(glob.glob('data/artificial/%s/train_*.tsv' % condition))

for fn in filenames:
  launch(fn)
