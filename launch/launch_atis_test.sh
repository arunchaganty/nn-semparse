#!/usr/bin/env bash
if [ -z $2 ]
then
  echo "Usage: $0 [hidden layer size] [word embedding size]" 1>&2
  exit 1
fi
d=$1
i=$2 
device=cpu
queue=jag
num_cpu=4
#flags='--request-docker-image codalab/theano-cuda7.0-352.39'
cl run :src :lib :evaluator :atis :atis-lexicon "OMP_NUM_THREADS=4 THEANO_FLAGS=blas.ldflags=-lopenblas,device=cpu,floatX=float32,cuda.root=/usr/local/cuda-7.0 python src/py/main.py -d $d -i $i -o $i -p none -u 1 -t 25,5,5 -c lstm -m encoderdecoder -k 10 --stats-file stats.json --train-data atis/atis_train.tsv --dev-data atis/atis_test.tsv --save-file params" --request-queue 'jag' --request-cpus 4 -n atis_test -d 'no augmentation'
sleep 1
cl run :src :lib :evaluator :atis :atis-lexicon "OMP_NUM_THREADS=4 THEANO_FLAGS=blas.ldflags=-lopenblas,device=cpu,floatX=float32,cuda.root=/usr/local/cuda-7.0 python src/py/main.py -d $d -i $i -o $i -p none -u 1 -t 25,5,5 -c lstm -m encoderdecoder -k 10 --stats-file stats.json --train-data atis-lexicon/atis_train_augmentLex.tsv --dev-data atis/atis_test.tsv --save-file params" --request-queue 'jag' --request-cpus 4 -n atis_test -d 'augment lex'
sleep 1
cl run :src :lib :evaluator :atis :atis-lexicon "OMP_NUM_THREADS=4 THEANO_FLAGS=blas.ldflags=-lopenblas,device=gpu3,floatX=float32,cuda.root=/usr/local/cuda-7.0 python src/py/main.py -d $d -i $i -o $i -p none -u 1 -t 25,5,5 -c lstm -m attention -k 10 --stats-file stats.json --train-data atis/atis_train.tsv --dev-data atis/atis_test.tsv --save-file params" --request-queue 'host=jagupard11' --request-cpus 1 -n atis_test -d 'no augmentation' --request-docker-image codalab/theano-cuda7.0-352.39
sleep 1
cl run :src :lib :evaluator :atis :atis-lexicon "OMP_NUM_THREADS=4 THEANO_FLAGS=blas.ldflags=-lopenblas,device=gpu2,floatX=float32,cuda.root=/usr/local/cuda-7.0 python src/py/main.py -d $d -i $i -o $i -p none -u 1 -t 25,5,5 -c lstm -m attention -k 10 --stats-file stats.json --train-data atis-lexicon/atis_train_augmentLex.tsv --dev-data atis/atis_test.tsv --save-file params" --request-queue 'host=jagupard11' --request-cpus 1 -n atis_test -d 'augment lex' --request-docker-image codalab/theano-cuda7.0-352.39 
sleep 1
cl run :src :lib :evaluator :atis :atis-lexicon "OMP_NUM_THREADS=4 THEANO_FLAGS=blas.ldflags=-lopenblas,device=gpu0,floatX=float32,cuda.root=/usr/local/cuda-7.0 python src/py/main.py -d $d -i $i -o $i -p attention -u 1 -t 25,5,5 -c lstm -m attention -k 10 --stats-file stats.json --train-data atis/atis_train.tsv --dev-data atis/atis_test.tsv --save-file params" --request-queue 'host=jagupard11' --request-cpus 1 -n atis_test -d 'no augmentation' --request-docker-image codalab/theano-cuda7.0-352.39
sleep 1
cl run :src :lib :evaluator :atis :atis-lexicon "OMP_NUM_THREADS=4 THEANO_FLAGS=blas.ldflags=-lopenblas,device=gpu1,floatX=float32,cuda.root=/usr/local/cuda-7.0 python src/py/main.py -d $d -i $i -o $i -p attention -u 1 -t 25,5,5 -c lstm -m attention -k 10 --stats-file stats.json --train-data atis-lexicon/atis_train_augmentLex.tsv --dev-data atis/atis_test.tsv --save-file params" --request-queue 'host=jagupard11' --request-cpus 1 -n atis_test -d 'augment lex' --request-docker-image codalab/theano-cuda7.0-352.39 
cl run :src :lib :evaluator :atis :atis-augmented "OMP_NUM_THREADS=4 THEANO_FLAGS=blas.ldflags=-lopenblas,device=cpu,floatX=float32,cuda.root=/usr/local/cuda-7.0 python src/py/main.py -d $d -i $i -o $i -p none -u 1 -t 25,5,5 -c lstm -m encoderdecoder -k 10 --stats-file stats.json --train-data atis-augmented/atis_train_augment2k.tsv --dev-data atis/atis_test.tsv --save-file params" --request-queue 'jag' --request-cpus 4 -n atis_test -d 'augment pcfg'
sleep 1
cl run :src :lib :evaluator :atis :atis-augmented "OMP_NUM_THREADS=4 THEANO_FLAGS=blas.ldflags=-lopenblas,device=gpu0,floatX=float32,cuda.root=/usr/local/cuda-7.0 python src/py/main.py -d $d -i $i -o $i -p none -u 1 -t 25,5,5 -c lstm -m attention -k 10 --stats-file stats.json --train-data atis-augmented/atis_train_augment2k.tsv --dev-data atis/atis_test.tsv --save-file params" --request-queue 'host=jagupard11' --request-cpus 1 -n atis_test -d 'augment pcfg' --request-docker-image codalab/theano-cuda7.0-352.39 
sleep 1
cl run :src :lib :evaluator :atis :atis-augmented "OMP_NUM_THREADS=4 THEANO_FLAGS=blas.ldflags=-lopenblas,device=gpu2,floatX=float32,cuda.root=/usr/local/cuda-7.0 python src/py/main.py -d $d -i $i -o $i -p attention -u 1 -t 25,5,5 -c lstm -m attention -k 10 --stats-file stats.json --train-data atis-augmented/atis_train_augment2k.tsv --dev-data atis/atis_test.tsv --save-file params" --request-queue 'host=jagupard11' --request-cpus 1 -n atis_test -d 'augment pcfg' --request-docker-image codalab/theano-cuda7.0-352.39 
