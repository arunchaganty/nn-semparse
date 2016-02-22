#!/usr/bin/env bash
if [ -z $2 ]
then
  echo "Usage: $0 [hidden layer size] [word embedding size]" 1>&2
  exit 1
fi
d=$1
i=$2 

# No augmentation
for seed in 0 1 2 3 4
do
  cl run :src :lib :evaluator :geo880 "OMP_NUM_THREADS=4 THEANO_FLAGS=blas.ldflags=-lopenblas python src/py/main.py -d $d -i $i -o $i -p attention -u 1 -t 20 -c lstm -m attention --stats-file stats.json --domain geoquery --dev-frac 0.2 --dev-seed ${seed} --model-seed ${seed} --train-data geo880/geo880_train600.tsv --save-file params" --request-docker-image robinjia/robinjia-codalab:1.0 --request-queue john --request-cpus 4 -n geo_train -d "no aug, ds ${seed}, ms ${seed}"
  sleep 1
done

# Augmentation
for aug in concat concat-mask single single-mask double double-mask pcfg
do
  for seed in 0 1 2 3 4
  do
    cl run :src :lib :evaluator :geo880 "OMP_NUM_THREADS=4 THEANO_FLAGS=blas.ldflags=-lopenblas python src/py/main.py -d $d -i $i -o $i -p attention -u 1 -t 20 -c lstm -m attention --stats-file stats.json --domain geoquery -a ${aug}:240 --dev-frac 0.2 --dev-seed ${seed} --model-seed ${seed} --train-data geo880/geo880_train600.tsv --save-file params" --request-docker-image robinjia/robinjia-codalab:1.0 --request-queue john --request-cpus 4 -n geo_train -d "${aug}:240, ds ${seed}, ms ${seed}"
    sleep 1
  done
done
