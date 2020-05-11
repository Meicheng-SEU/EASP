# EASP
An Explainable Artificial Intelligence Predictor for Early Detection of Sepsis

## Brief Introduction
The PhysioNet/Computing in Cardiology Challenge 2019 facilitated the development of automated, open-source algorithms for the early detection of sepsis from clinical data. Details see (https://physionet.org/content/challenge-2019/1.0.0/).

We proposed an Explainable Artificial-intelligence Sepsis Predictor (EASP) to predict sepsis risk hour-by-hour, and focused on its interpretability for the clinical EHR data sourced from ICU patients. Final results show that EASP achieved best performance in the challenge.

## Data
These instructions go through the training and evaluation of our model on the Physionet 2019 challenge public database (https://archive.physionet.org/users/shared/challenge-2019/).

To download and build the datasets run:

    ./setup.sh

## Training
To train a model use the following command:

    python model_train.py
  
Note that the model is saved in directory of 'xgb_model'

## Evaluation
After training the model, you can make predictions and then yield the model performance.

    python test.py xgb_model
    
Or you can directly use our trained model for quick verification using the following command
  
    python test.py Submit_model
    
## Explanation
Impacts of features on risk output were quantified by Shapley values to obtain instant interpretability for the developed EASP model.

    python shap_explain.py xgb_model  
    or  
    python shap_explain.py Submit_model

## Citation and Reference
This work is published in the following paper in Critical Care Medicine

    An Explainable Artificial Intelligence Predictor for Early Detection of Sepsis
