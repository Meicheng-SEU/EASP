# EASP
An Explainable Artificial Intelligence Predictor for Early Detection of Sepsis

## Brief Introduction
The PhysioNet/Computing in Cardiology Challenge 2019 facilitated the development of automated, open-source algorithms for the early detection of sepsis from clinical data. Details see (https://physionet.org/content/challenge-2019/1.0.0/)

We proposed an Explainable Artificial-intelligence Sepsis Predictor (EASP) to predict sepsis risk hour-by-hour, and focused on its interpretability for the clinical EHR data sourced from ICU patients. 

## Data
These instructions go through the training and evaluation of our model on the Physionet 2019 challenge public database (https://archive.physionet.org/users/shared/challenge-2019/).

To download and build the datasets run:

  ./setup.sh

## Training
To train a model use the following command:

  python model_train.py
  
Note that the model is saved in directory of ./xgb_model

## Evaluation
After training the model, you can make predictions and then yeild the model performance.

  python test.py
  
## Citation and Reference
This work is published in the following paper in Critical Care Medicine

An Explainable Artificial Intelligence Predictor for Early Detection of Sepsis
