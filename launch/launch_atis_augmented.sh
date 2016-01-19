#!/usr/bin/env bash
if [ -z $2 ]
then
  echo "Usage: $0 [hidden layer size] [word embedding size]" 1>&2
  exit 1
fi
d=$1
i=$2 
device=gpu2
queue='host=jagupard11'
num_cpu=1
flags='--request-docker-image codalab/theano-cuda7.0-352.39'
cl run :src :lib :evaluator :atis :atis-lexicon "OMP_NUM_THREADS=4 THEANO_FLAGS=blas.ldflags=-lopenblas,device=${device},floatX=float32,cuda.root=/usr/local/cuda-7.0 python src/py/main.py -d $d -i $i -o $i -p attention -u 1 -t 25,5,5 -c lstm -m attention --stats-file stats.json --train-data atis-lexicon/atis_train_augmentLex.tsv --dev-data atis/atis_dev.tsv --save-file params" --request-queue $queue --request-cpus $num_cpu -n atis_tune -d 'augment lex' $flags
