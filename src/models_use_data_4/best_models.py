# -*- coding=utf-8 -*-

# 调用函数获得最佳参数的模型

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
import pandas as pd
import numpy as np
import getpass


# ----------当前最优参数----------------
def get_tuned_xgb():
    from xgboost.sklearn import XGBClassifier
    return XGBClassifier(
        learning_rate=0.03,  # eta
        n_estimators=860,
        max_depth=3,
        min_child_weight=1,
        gamma=0.1,
        subsample=0.96,
        colsample_bytree=0.94,
        reg_alpha=1e-7,  # alpha
        reg_lambda=6,  # lambda
        objective='binary:logistic',
        nthread=-1,
        seed=36)


def get_tuned_gbm():
    return GradientBoostingClassifier(
        learning_rate=0.1,
        n_estimators=256,
        max_depth=3,
        min_samples_split=400,
        min_samples_leaf=30,
        random_state=36)


def get_tuned_rf():
    return RandomForestClassifier(
        n_estimators=400,
        max_features='sqrt',
        max_depth=19,
        min_samples_split=5,
        min_samples_leaf=18,
        random_state=36,
        n_jobs=-1)


def get_tuned_lr():
    return LogisticRegression(
        penalty='l2',
        C=0.1,
        random_state=36)


def get_tuned_svc():
    svc =  SVC(
        C=10,
        gamma='auto',
        random_state=36)
    return CalibratedClassifierCV(svc, cv=5, method='sigmoid')


def get_data(train_df, test_df ,DATA_DIR, model_name):
    target = 'target'
    print('选择{}特征...'.format(model_name))
    print('位置：', DATA_DIR)

    # 使用特征选择的结果
    support = pd.read_csv(DATA_DIR + 'selection_result_' + model_name + '.csv', encoding='gbk',
                          header=None, names=['var', 'is_useful'])
    predictors = support[support['is_useful'] == True]['var'].values

    train_df = train_df[predictors].join(train_df[target])

    x_train = np.array(train_df[predictors])
    y_train = train_df[target].ravel()
    x_test = np.array(test_df[predictors])

    return x_train, y_train, x_test
