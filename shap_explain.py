import xgboost as xgb
import shap
import numpy as np, os, sys
from feature_engineering import data_process

def shap_value(input_data, k_fold, model_path):
    shap.initjs()
    all_shap_values = np.zeros((input_data.shape[0], input_data.shape[1]))
    dat = xgb.DMatrix(input_data)
    for k in range(k_fold):
        file_name = './' + model_path + '/' + 'model{}.mdl'.format(k + 1)
        xgb_model = xgb.Booster(model_file = file_name)
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(dat)
        all_shap_values = all_shap_values + shap_values

    return all_shap_values / 5

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise Exception('Include the model directory as arguments, '
                        'e.g., python shap_explain.py Submit_model')

    data_path = "./data/all_dataset/"
    train_nosepsis = np.load('./data/train_nosepsis.npy')
    train_sepsis = np.load('./data/train_sepsis.npy')

    train_set = np.append(train_sepsis, train_nosepsis)
    features, labels = data_process(train_set, data_path)

    xgb_model_path = sys.argv[1]
    shap_data = shap_value(features, k_fold = 5, model_path = xgb_model_path)
    shap.summary_plot(shap_data, features, max_display = 20, plot_type = "bar")
    shap.summary_plot(shap_data, features, max_display = 20, plot_type = "dot")
