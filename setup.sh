#!/bin/bash

url=https://archive.physionet.org/users/shared/challenge-2019/

mkdir data prediction label xgb_model

cd data
mkdir all_dataset

curl -O $url/training_setA.zip
unzip training_setA.zip
curl -O $url/training_setB.zip
unzip training_setB.zip

cd ..

python build_datasets.py
