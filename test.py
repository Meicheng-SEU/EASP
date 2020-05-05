import os
import pandas as pd
import numpy as np
import xgboost as xgb
from evaluate_sepsis_score import evaluate_sepsis_score
from feature_engineering import feature_extraction

def save_challenge_predictions(file, scores, labels):
    with open(file, 'w') as f:
        f.write('PredictedProbability|PredictedLabel\n')
        for (s, l) in zip(scores, labels):
            f.write('%g|%d\n' % (s, l))

def save_challenge_testlabel(file, labels):
    with open(file, 'w') as f:
        f.write('SepsisLabel\n')
        for l in labels:
            f.write('%d\n' % l)

def load_model_predict(X_test, k_fold, path):
    test_pred = np.zeros((X_test.shape[0], k_fold))
    X_test = xgb.DMatrix(X_test)
    for k in range(k_fold):
        model_path_name = path + '/model{}.mdl'.format(k+1)
        xgb_model = xgb.Booster(model_file = model_path_name)
        y_test_pred = xgb_model.predict(X_test)
        test_pred[:, k] = y_test_pred
    test_pred = pd.DataFrame(test_pred)
    result_pro = test_pred.mean(axis=1)

    return result_pro

def predict(data_set, data_dir, save_prediction_dir,
            save_label_dir, model_path, risk_threshold):
    for psv in data_set:
        patient = pd.read_csv(os.path.join(data_dir, psv), sep='|')
        features, labels = feature_extraction(patient)

        predict_pro = load_model_predict(features, k_fold = 5, path = model_path)
        PredictedProbability = np.array(predict_pro)
        PredictedLabel = [0 if i <= risk_threshold else 1 for i in predict_pro]

        save_prediction_name = save_prediction_dir + psv
        save_challenge_predictions(save_prediction_name, PredictedProbability, PredictedLabel)
        save_testlabel_name = save_label_dir + psv
        save_challenge_testlabel(save_testlabel_name, labels)

if __name__ == "__main__":
    #data_path = "data/all_dataset/"
    test_set = np.load('test_set.npy')
    test_data_path_dir = 'E:/Cinc2019/Version2/cinc2019/version2_7_17/training/'
    prediction_directory = 'prediction/'
    label_directory = 'test_label/'
    model_path = 'xgboost_model'

    predict(test_set, test_data_path_dir, prediction_directory, label_directory, model_path, 0.525)
    # score = auroc, auprc, accuracy, f_measure, normalized_observed_utility
    evaluate_sepsis_score(label_directory, prediction_directory)