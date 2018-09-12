# -*- coding=utf-8 -*-

# 调用函数获得最佳参数的模型

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV


# ----------当前最优参数----------------
def get_tuned_xgb():
    from xgboost.sklearn import XGBClassifier
    return XGBClassifier(
        learning_rate=0.01,  # eta
        n_estimators=5000,
        max_depth=3,
        min_child_weight=3,
        gamma=0,
        subsample=1,
        colsample_bytree=0.6,
        reg_alpha=0.15,  # alpha
        reg_lambda=10,  # lambda
        objective='binary:logistic',
        nthread=-1,
        seed=36)


def get_tuned_gbm():
    return GradientBoostingClassifier(
        learning_rate=0.01,
        n_estimators=3000,
        max_depth=4,
        min_samples_split=2,
        min_samples_leaf=15,
        max_features=None,
        subsample=1.0,
        random_state=36)

def get_tuned_rf():
    return RandomForestClassifier(
        n_estimators=3000,
        max_features='sqrt',
        max_depth=11,
        min_samples_split=2,
        min_samples_leaf=17,
        random_state=36,
        n_jobs=-1)

def get_tuned_lr():
    return LogisticRegression(
        penalty='l1',
        C=0.1,
        random_state=36)

def get_tuned_svc():
    svc =  SVC(
        C=1,
        gamma='auto',
        kernel='rbf',
        random_state=36)
    return CalibratedClassifierCV(svc, cv=5, method='sigmoid')

