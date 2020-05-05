#!/bin/bash

url=https://archive.physionet.org/users/shared/challenge-2019/

mkdir data && cd data
# how to build a new die
curl -O $url/training_setA.zip
unzip training_setA.zip
curl -O $url/training_setB.zip
unzip training_setA.zip

cd ..

python build_datasets.py

