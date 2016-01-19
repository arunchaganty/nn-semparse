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
cl run :src :lib :evaluator :geo880-lesscopy :geo880-augmented "OMP_NUM_THREADS=4 THEANO_FLAGS=blas.ldflags=-lopenblas,device=${device},floatX=float32,cuda.root=/usr/local/cuda-7.0 python src/py/main.py -d $d -i $i -o $i -p attention -u 1 -t 25,5,5 -c lstm -m attention -k 10 --stats-file stats.json --domain geoquery --train-data geo880-lesscopy/geo880_train500.tsv --dev-data geo880-lesscopy/geo880_dev100.tsv --save-file params" --request-queue jag --request-cpus $num_cpu -n geoquery_tune -d 'no augmentation' $flags
sleep 1
cl run :src :lib :evaluator :geo880-lesscopy :geo880-augmented "OMP_NUM_THREADS=4 THEANO_FLAGS=blas.ldflags=-lopenblas,device=${device},floatX=float32,cuda.root=/usr/local/cuda-7.0 python src/py/main.py -d $d -i $i -o $i -p attention -u 1 -t 25,5,5 -c lstm -m attention -k 10 --stats-file stats.json --domain geoquery --train-data geo880-augmented/geo880_train500_augpcfg250.tsv --dev-data geo880-lesscopy/geo880_dev100.tsv --save-file params" --request-queue jag --request-cpus $num_cpu -n geoquery_tune -d 'augment 250' $flags
sleep 1
cl run :src :lib :evaluator :geo880-lesscopy :geo880-augmented "OMP_NUM_THREADS=4 THEANO_FLAGS=blas.ldflags=-lopenblas,device=${device},floatX=float32,cuda.root=/usr/local/cuda-7.0 python src/py/main.py -d $d -i $i -o $i -p attention -u 1 -t 25,5,5 -c lstm -m attention -k 10 --stats-file stats.json --domain geoquery --train-data geo880-augmented/geo880_train500_augpcfg500.tsv --dev-data geo880-lesscopy/geo880_dev100.tsv --save-file params" --request-queue jag --request-cpus $num_cpu -n geoquery_tune -d 'augment 500' $flags
sleep 1
cl run :src :lib :evaluator :geo880-lesscopy :geo880-augmented "OMP_NUM_THREADS=4 THEANO_FLAGS=blas.ldflags=-lopenblas,device=${device},floatX=float32,cuda.root=/usr/local/cuda-7.0 python src/py/main.py -d $d -i $i -o $i -p attention -u 1 -t 25,5,5 -c lstm -m attention -k 10 --stats-file stats.json --domain geoquery --train-data geo880-augmented/geo880_train500_augpcfg750.tsv --dev-data geo880-lesscopy/geo880_dev100.tsv --save-file params" --request-queue jag --request-cpus $num_cpu -n geoquery_tune -d 'augment 750' $flags
sleep 1
cl run :src :lib :evaluator :geo880-lesscopy :geo880-augmented "OMP_NUM_THREADS=4 THEANO_FLAGS=blas.ldflags=-lopenblas,device=${device},floatX=float32,cuda.root=/usr/local/cuda-7.0 python src/py/main.py -d $d -i $i -o $i -p attention -u 1 -t 25,5,5 -c lstm -m attention -k 10 --stats-file stats.json --domain geoquery --train-data geo880-augmented/geo880_train500_augpcfg1k.tsv --dev-data geo880-lesscopy/geo880_dev100.tsv --save-file params" --request-queue jag --request-cpus $num_cpu -n geoquery_tune -d 'augment 1k' $flags
sleep 1
cl run :src :lib :evaluator :geo880-lesscopy :geo880-augmented "OMP_NUM_THREADS=4 THEANO_FLAGS=blas.ldflags=-lopenblas,device=${device},floatX=float32,cuda.root=/usr/local/cuda-7.0 python src/py/main.py -d $d -i $i -o $i -p attention -u 1 -t 25,5,5 -c lstm -m attention -k 10 --stats-file stats.json --domain geoquery --train-data geo880-augmented/geo880_train500_augpcfg1500.tsv --dev-data geo880-lesscopy/geo880_dev100.tsv --save-file params" --request-queue jag --request-cpus $num_cpu -n geoquery_tune -d 'augment 1500' $flags
sleep 1
cl run :src :lib :evaluator :geo880-lesscopy :geo880-augmented "OMP_NUM_THREADS=4 THEANO_FLAGS=blas.ldflags=-lopenblas,device=${device},floatX=float32,cuda.root=/usr/local/cuda-7.0 python src/py/main.py -d $d -i $i -o $i -p attention -u 1 -t 25,5,5 -c lstm -m attention -k 10 --stats-file stats.json --domain geoquery --train-data geo880-augmented/geo880_train500_augpcfg2k.tsv --dev-data geo880-lesscopy/geo880_dev100.tsv --save-file params" --request-queue jag --request-cpus $num_cpu -n geoquery_tune -d 'augment 2k' $flags
