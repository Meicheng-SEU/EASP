# EASP
An Explainable Artificial Intelligence Predictor for Early Detection of Sepsis. The highest entry score from ***SailOcean*** in the PhysioNet/Computing in Cardiology Challenge 2019.

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
    
Or you can directly use our trained model for quick verification using the following command.
  
    python test.py Submit_model
    
## Explanation
Impacts of features on risk output were quantified by Shapley values to obtain instant interpretability for the developed EASP model.

    python shap_explain.py xgb_model  
    or  
    python shap_explain.py Submit_model

## Citation and Reference
This work has been published in ***Critical Care Medicine***.

    [An Explainable Artificial Intelligence Predictor for Early Detection of Sepsis]
    (https://journals.lww.com/ccmjournal/Fulltext/2020/11000/An_Explainable_Artificial_Intelligence_Predictor.37.aspx)
    
Conference Paper published in ***2019 Computing in Cardiology Conference*** is as follows.

   [Early Prediction of Sepsis Using Multi-Feature Fusion Based XGBoost Learning and Bayesian Optimization](https://www.researchgate.net/publication/338628580_Early_Prediction_of_Sepsis_Using_Multi-Feature_Fusion_Based_XGBoost_Learning_and_Bayesian_Optimization)
    
## Feadback
If you have any questions or suggestions on this work, please e-mail meicheng@seu.edu.cn
