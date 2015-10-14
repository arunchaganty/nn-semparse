# Neural Network Semantic Parsing

Example usage:

    python src/py/main.py -d 80 -i 20 -o 20 -t 10 -c lstm -m encoderdecoder --train-data data/overnight/processed/publications_train.tsv --dev-data data/overnight/processed/publications_dev.tsv

Warning: some flags probably don't work right now.  
--save-file and --load-file do work, and are useful to store learned parameters
and load them back later for evaluation.
