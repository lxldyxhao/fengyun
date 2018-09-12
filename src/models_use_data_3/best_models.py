# -*- coding=utf-8 -*-

# 调用函数获得最佳参数的模型

from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV

# ----------当前最优参数----------------
def get_tuned_xgb():
    from xgboost.sklearn import XGBClassifier
    return XGBClassifier(
        learning_rate=0.005,
        n_estimators=2660,
        max_depth=5,
        min_child_weight=5,
        gamma=0.7,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0,
        reg_lambda=1,
        objective='binary:logistic',
        nthread=-1,
        seed=36)

def get_tuned_gbm():
    return GradientBoostingClassifier(
        learning_rate=0.015,
        n_estimators=1600,
        max_depth=5,
        min_samples_split=550,
        min_samples_leaf=48,
        max_features=27,
        subsample=0.8,
        random_state=36)

def get_tuned_rf():
    return RandomForestClassifier(
        n_estimators=400,
        max_features=300,
        max_depth=16,
        min_samples_split=30,
        min_samples_leaf=15,
        random_state=36,
        n_jobs=-1)

def get_tuned_sgd_lr():
    return SGDClassifier(
        loss='log',
        penalty='elasticnet',
        learning_rate='invscaling',
        alpha=0.001,
        l1_ratio=0.55,
        eta0=0.15,
        shuffle=True,
        random_state=36)

def get_tuned_sgd_svm():
    svm =  SGDClassifier(
        loss='hinge',
        penalty='elasticnet',
        alpha=0.001,
        learning_rate='invscaling',
        l1_ratio=0.55,
        eta0=0.07,
        shuffle=True,
        random_state=36)
    return CalibratedClassifierCV(svm, cv=1, method='sigmoid')

