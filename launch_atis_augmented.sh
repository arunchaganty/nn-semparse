#!/usr/bin/env bash
if [ -z $2 ]
then
  echo "Usage: $0 [hidden layer size] [word embedding size]" 1>&2
  exit 1
fi
d=$1
i=$2 
device1=gpu1
device2=gpu3
queue1='host=jagupard7'
queue2='host=jagupard11'
num_cpu=1
flags='--request-docker-image codalab/theano-cuda7.0-352.39'
cl run :src :lib :evaluator :atis :atis_augmented "OMP_NUM_THREADS=4 THEANO_FLAGS=blas.ldflags=-lopenblas,device=${device1},floatX=float32,cuda.root=/usr/local/cuda-7.0 python src/py/main.py -d $d -i $i -o $i -p attention -u 1 -t 25,5,5 -c lstm -m attention --stats-file stats.json --train-data atis_augmented/atis_train_augment2k.tsv --dev-data atis/atis_dev.tsv --save-file params" --request-queue $queue1 --request-cpus $num_cpu -n atis_tune -d 'augment 2k' $flags
sleep 5
cl run :src :lib :evaluator :atis :atis_augmented "OMP_NUM_THREADS=4 THEANO_FLAGS=blas.ldflags=-lopenblas,device=${device2},floatX=float32,cuda.root=/usr/local/cuda-7.0 python src/py/main.py -d $d -i $i -o $i -p attention -u 1 -t 25,5,5 -c lstm -m attention --stats-file stats.json --train-data atis_augmented/atis_train_augment4k.tsv --dev-data atis/atis_dev.tsv --save-file params" --request-queue $queue2 --request-cpus $num_cpu -n atis_tune -d 'augment 4k' $flags
