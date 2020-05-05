import xgboost as xgb
import shap
import numpy as np
import pandas as pd

def load_model_shap(X_test, k_fold, path):  # 加载训练好的模型
    shap.initjs()
    expected_value = np.zeros((k_fold, 1))
    test_pred = np.zeros((X_test.shape[0], k_fold))
    all_shap_values = np.zeros((X_test.shape[0], X_test.shape[1]))
    X_test = xgb.DMatrix(X_test)
    for k in range(k_fold):
        model_path = path + '/model{}.mdl'.format(k + 1)
        xgb_model = xgb.Booster(model_file=model_path)
        # y_test_pred = xgb_model.predict(X_test)
        # test_pred[:, k] = y_test_pred

        explainer = shap.TreeExplainer(xgb_model)
        expected_value[k] = explainer.expected_value
        shap_values = explainer.shap_values(X_test)
        all_shap_values = all_shap_values + shap_values

    # test_pred = pd.DataFrame(test_pred)
    # result_pro = test_pred.mean(axis=1)
    expected_value = np.mean(expected_value)

    return (all_shap_values / 5), expected_value