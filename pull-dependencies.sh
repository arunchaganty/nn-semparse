#!/bin/bash
if [ ! -d refdb ]
then
  git clone https://github.com/percyliang/refdb.git
  cd refdb
  ./generate.rb bib out=all.bib
  cd ../tex
  ln -s ../refdb/all.bib
  cd ..
fi
