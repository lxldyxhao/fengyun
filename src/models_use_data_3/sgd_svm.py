# -*- coding=utf-8 -*-

# 用sklearn的 Stochastic Gradient Descent 训练，

print('导入算法包...')
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn import model_selection, metrics
import sklearn
from sklearn.model_selection import GridSearchCV
import time
import getpass
time_begin = time.time()

print('读取数据...')
train_df = pd.read_csv('../../data/data_3/train_preprocessed1.csv', encoding='gbk')
target = 'target'
predictors = train_df.columns.drop(['id','target'])

# 不同的操作系统，设置不同的数据量和线程数
n_thread = 4
print_importance=False
if getpass.getuser() == 'stone':
    # train_df = train_df[:2000]
    n_thread = 2
    print_importance = True
elif getpass.getuser() == 'lxl':
    n_thread = 2
print('数据量：', train_df.shape)

# ------------交叉验证函数----------
def modelfit(model, dtrain, useTrainCV=True, cv_fold=5, print_importance=False):
    # fit
    model.fit(dtrain[predictors], dtrain[target])
    # predict
    dtrain_predictions = model.predict(dtrain[predictors])
    dtrain_predprob = model.predict_proba(dtrain[predictors])[:, 1]
    # print model report
    print('\nModel Report')
    print('Accuracy:%f' % sklearn.metrics.accuracy_score(dtrain[target].values, dtrain_predictions))
    print('AUC Score (Train): %f' % sklearn.metrics.roc_auc_score(dtrain[target], dtrain_predprob))
    if useTrainCV:
        print('CV...')
        cv_score = model_selection.cross_val_score(model, dtrain[predictors], dtrain[target],
                                                   cv=cv_fold, scoring='roc_auc')
        print('CV Score roc_auc: Mean - %f | Std - %f | Min - %f | Max - %f' %
              (np.mean(cv_score), np.std(cv_score), np.min(cv_score), np.max(cv_score)))

# ----------当前最优参数----------------
def get_tuned_sgd_svm():
    return SGDClassifier(
        loss='hinge',
        penalty='elasticnet',
        alpha=0.001,
        learning_rate='invscaling',
        l1_ratio=0.55,
        eta0=0.07,
        shuffle=True,
        random_state=36)

sgd_svm_model = get_tuned_sgd_svm()

# -----------只用当前参数训练一个模型-------------
# print('训练模型中...')
#
# modelfit(lr_model, train_df, print_importance=False)

# -----------END 只用当前参数训练一个模型---------


# ---------------格点搜索----------------------
param_test = {

}

gsearch = GridSearchCV(
    estimator=sgd_svm_model,
    param_grid=param_test,
    scoring='roc_auc',
    n_jobs=n_thread,
    cv=5
)

print('正在搜索...')
gsearch.fit(train_df[predictors], train_df[target])

# 输出搜索结果
print('\n当前格点搜索的参数：\n', param_test)
print('\ngsearch.best_params_:', gsearch.best_params_, )
print('\ngsearch.best_score_:', gsearch.best_score_, )
# ---------------END 格点搜索----------------

# ------------输出运行时间　不需要改---------------
time_spend = time.time() - time_begin
print('\n运行时间：%d 秒，约%d分钟\n' % (time_spend, time_spend // 60))

# todo: 使用模型对test集合进行预测

# todo: 存储调好参数的模型
