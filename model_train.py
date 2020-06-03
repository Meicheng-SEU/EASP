import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, accuracy_score
import xgboost as xgb
from hyperopt import STATUS_OK, hp, fmin, tpe
from feature_engineering import data_process

def BO_TPE(X_train, y_train, X_val, y_val):
    "Hyperparameter optimization"
    train = xgb.DMatrix(X_train, label=y_train)
    val = xgb.DMatrix(X_val, label=y_val)
    X_val_D = xgb.DMatrix(X_val)

    def objective(params):
        xgb_model = xgb.train(params, dtrain=train, num_boost_round=1000, evals=[(val, 'eval')],
                              verbose_eval=False, early_stopping_rounds=80)
        y_vd_pred = xgb_model.predict(X_val_D, ntree_limit=xgb_model.best_ntree_limit)
        y_val_class = [0 if i <= 0.5 else 1 for i in y_vd_pred]

        acc = accuracy_score(y_val, y_val_class)
        loss = 1 - acc

        return {'loss': loss, 'params': params, 'status': STATUS_OK}

    max_depths = [3, 4]
    learning_rates = [0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2]
    subsamples = [0.5, 0.6, 0.7, 0.8, 0.9]
    colsample_bytrees = [0.5, 0.6, 0.7, 0.8, 0.9]
    reg_alphas = [0.0, 0.005, 0.01, 0.05, 0.1]
    reg_lambdas = [0.8, 1, 1.5, 2, 4]

    space = {
        'max_depth': hp.choice('max_depth', max_depths),
        'learning_rate': hp.choice('learning_rate', learning_rates),
        'subsample': hp.choice('subsample', subsamples),
        'colsample_bytree': hp.choice('colsample_bytree', colsample_bytrees),
        'reg_alpha': hp.choice('reg_alpha', reg_alphas),
        'reg_lambda': hp.choice('reg_lambda', reg_lambdas),
    }

    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=20)

    best_param = {'max_depth': max_depths[(best['max_depth'])],
                  'learning_rate': learning_rates[(best['learning_rate'])],
                  'subsample': subsamples[(best['subsample'])],
                  'colsample_bytree': colsample_bytrees[(best['colsample_bytree'])],
                  'reg_alpha': reg_alphas[(best['reg_alpha'])],
                  'reg_lambda': reg_lambdas[(best['reg_lambda'])]
                  }

    return best_param

def train_model(k, X_train, y_train, X_val, y_val, save_model_dir):
    print('*************************************************************')
    print('{}th training ..............'.format(k + 1))
    print('Hyperparameters optimization')
    best_param = BO_TPE(X_train, y_train, X_val, y_val)
    xgb_model = xgb.XGBClassifier(max_depth = best_param['max_depth'],
                                  eta = best_param['learning_rate'],
                                  n_estimators = 1000,
                                  subsample = best_param['subsample'],
                                  colsample_bytree = best_param['colsample_bytree'],
                                  reg_alpha = best_param['reg_alpha'],
                                  reg_lambda = best_param['reg_lambda'],
                                  objective = "binary:logistic"
                                  )

    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='error',
                  early_stopping_rounds=80, verbose=False)

    y_tr_pred = (xgb_model.predict_proba(X_train, ntree_limit=xgb_model.best_ntree_limit))[:, 1]
    train_auc = roc_auc_score(y_train, y_tr_pred)
    print('training dataset AUC: ' + str(train_auc))
    y_tr_class = [0 if i <= 0.5 else 1 for i in y_tr_pred]
    acc = accuracy_score(y_train, y_tr_class)
    print('training dataset acc: ' + str(acc))

    y_vd_pred = (xgb_model.predict_proba(X_val, ntree_limit=xgb_model.best_ntree_limit))[:, 1]
    valid_auc = roc_auc_score(y_val, y_vd_pred)
    print('validation dataset AUC: ' + str(valid_auc))
    y_val_class = [0 if i <= 0.5 else 1 for i in y_vd_pred]
    acc = accuracy_score(y_val, y_val_class)
    print('validation dataset acc: ' + str(acc))
    print('************************************************************')
    # save the model
    save_model_path = save_model_dir + 'model{}.mdl'.format(k + 1)
    xgb_model.get_booster().save_model(fname=save_model_path)

def downsample(data_set, data_dir):
    """
    Using our feature extraction approach will result in over 1 million hours of data in the training process.
    However, only roughly 1.8% of these data corresponds to a positive outcome.
    Consequently, in order to deal with the serious class imbalance, a systematic way is provided by
    down sampling the excessive data instances of the majority class in each cross validation.
    """
    x, y = data_process(data_set, data_dir)
    index_0 = np.where(y == 0)[0]
    index_1 = np.where(y == 1)[0]

    index = index_0[len(index_1): -1]
    x_del = np.delete(x, index, 0)
    y_del = np.delete(y, index, 0)
    index = [i for i in range(len(y_del))]
    np.random.shuffle(index)
    x_del = x_del[index]
    y_del = y_del[index]

    return x_del, y_del

if __name__ == "__main__":
    data_path = "./data/all_dataset/"
    train_nosepsis = np.load('./data/train_nosepsis.npy')
    train_sepsis = np.load('./data/train_sepsis.npy')

    # 5-fold cross validation was implemented and five XGBoost models were produced
    kfold = KFold(n_splits=5, shuffle=True, random_state=np.random.seed(12306))
    for (k, (train0_index, val0_index)), (k, (train1_index, val1_index)) in \
            zip(enumerate(kfold.split(train_nosepsis)), enumerate(kfold.split(train_sepsis))):

        train_set = np.append(train_nosepsis[train0_index], train_sepsis[train1_index])
        x_train, y_train = downsample(train_set, data_path)

        val_set = np.append(train_nosepsis[val0_index], train_sepsis[val1_index])
        x_val, y_val = downsample(val_set, data_path)

        train_model(k, x_train, y_train, x_val, y_val, save_model_dir = './xgb_model/')
