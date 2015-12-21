#!/usr/bin/env bash
if [ -z $2 ]
then
  echo "Usage: $0 [hidden layer size] [word embedding size]" 1>&2
  exit 1
fi
d=$1
i=$2 
device=cpu
num_cpu=4
#flags='--request-docker-image codalab/theano-cuda7.0-352.39'
cl run :src :lib :evaluator :geo880-lesscopy :geo880-augmented "OMP_NUM_THREADS=4 THEANO_FLAGS=blas.ldflags=-lopenblas,device=${device},floatX=float32,cuda.root=/usr/local/cuda-7.0 python src/py/main.py -d $d -i $i -o $i -p attention -u 1 -t 15,5,5 -c lstm -m attention --stats-file stats.json --domain geoquery --train-data geo880-lesscopy/geo880_train500.tsv --dev-data geo880-lesscopy/geo880_dev100.tsv --save-file params" --request-queue jag --request-cpus $num_cpu -n geoquery_tune -d 'no augmentation' $flags
sleep 5
cl run :src :lib :evaluator :geo880-lesscopy :geo880-augmented "OMP_NUM_THREADS=4 THEANO_FLAGS=blas.ldflags=-lopenblas,device=${device},floatX=float32,cuda.root=/usr/local/cuda-7.0 python src/py/main.py -d $d -i $i -o $i -p attention -u 1 -t 15,5,5 -c lstm -m attention --stats-file stats.json --domain geoquery --train-data geo880-augmented/geo880_train500_augment500.tsv --dev-data geo880-lesscopy/geo880_dev100.tsv --save-file params" --request-queue jag --request-cpus $num_cpu -n geoquery_tune -d 'augment 500' $flags
sleep 5
cl run :src :lib :evaluator :geo880-lesscopy :geo880-augmented "OMP_NUM_THREADS=4 THEANO_FLAGS=blas.ldflags=-lopenblas,device=${device},floatX=float32,cuda.root=/usr/local/cuda-7.0 python src/py/main.py -d $d -i $i -o $i -p attention -u 1 -t 15,5,5 -c lstm -m attention --stats-file stats.json --domain geoquery --train-data geo880-augmented/geo880_train500_augment1k.tsv --dev-data geo880-lesscopy/geo880_dev100.tsv --save-file params" --request-queue jag --request-cpus $num_cpu -n geoquery_tune -d 'augment 1k' $flags
sleep 5
cl run :src :lib :evaluator :geo880-lesscopy :geo880-augmented "OMP_NUM_THREADS=4 THEANO_FLAGS=blas.ldflags=-lopenblas,device=${device},floatX=float32,cuda.root=/usr/local/cuda-7.0 python src/py/main.py -d $d -i $i -o $i -p attention -u 1 -t 15,5,5 -c lstm -m attention --stats-file stats.json --domain geoquery --train-data geo880-augmented/geo880_train500_augment2k.tsv --dev-data geo880-lesscopy/geo880_dev100.tsv --save-file params" --request-queue jag --request-cpus $num_cpu -n geoquery_tune -d 'augment 2k' $flags
sleep 5
cl run :src :lib :evaluator :geo880-lesscopy :geo880-augmented "OMP_NUM_THREADS=4 THEANO_FLAGS=blas.ldflags=-lopenblas,device=${device},floatX=float32,cuda.root=/usr/local/cuda-7.0 python src/py/main.py -d $d -i $i -o $i -p attention -u 1 -t 15,5,5 -c lstm -m attention --stats-file stats.json --domain geoquery --train-data geo880-augmented/geo880_train500_augment4k.tsv --dev-data geo880-lesscopy/geo880_dev100.tsv --save-file params" --request-queue jag --request-cpus $num_cpu -n geoquery_tune -d 'augment 4k' $flags
