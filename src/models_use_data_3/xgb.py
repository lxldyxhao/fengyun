# -*- coding=utf-8 -*-

# 用xgboost训练，调参后的模型存放在../models中
# 命名： [数据名]_xgb
# 比如： data_3_xgb

print('导入算法包...')
import time
import getpass
import pickle
import pandas as pd
import xgboost as xgb
import sklearn
from sklearn.model_selection import GridSearchCV
from xgboost.sklearn import XGBClassifier
time_begin = time.time()

# ---------------读取数据---------------------
print('读取数据...')
train_df = pd.read_csv('../../data/data_3/train_preprocessed1.csv', encoding='gbk')
# test_df = pd.read_csv('../data/data_3/test_preprocessed1.csv', encoding='gbk')
target = 'target'
predictors = train_df.columns.drop(['id','target'])
# --------------END 读取数据------------------

# -----对操作系统，设置不同的数据量和线程数---------
n_thread = 4
print_importance=False
if getpass.getuser() == 'stone':
    n_thread = 2
    print_importance = True
elif getpass.getuser() == 'lxl':
    n_thread = 4
print('数据量：', train_df.shape)
# --------------END 操作系统----------------------

def modelfit(model, dtrain, useTrainCV=True, cv_fold=5, early_stopping_rounds=50):
    if useTrainCV:
        print('交叉验证自动设定迭代次数...')
        xgb_param = model.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        # cv:获取最优的boost_round?
        cvresult = xgb.cv(xgb_param, xgtrain,
                          num_boost_round=xgb_param['n_estimators'],
                          nfold=cv_fold,
                          metrics='auc',
                          early_stopping_rounds=early_stopping_rounds,
                          seed=36,
                          callbacks=[xgb.callback.print_evaluation(show_stdv=True)])
        print('cv完成 n_estimators：', cvresult.shape[0])
        model.set_params(n_estimators=cvresult.shape[0])
    # fit
    print('模型拟合中...')
    model.fit(dtrain[predictors],
            dtrain[target],
            eval_metric='auc',
            verbose=True)# todo
    # predict
    dtrain_predictions = model.predict(dtrain[predictors])
    dtrain_predprob = model.predict_proba(dtrain[predictors])[:, 1]
    # print model report
    print('\nModel Report')
    print('Accuracy:%f' % sklearn.metrics.accuracy_score(dtrain[target].values, dtrain_predictions))
    print('AUC Score (Train): %f' % sklearn.metrics.roc_auc_score(dtrain[target], dtrain_predprob))
    # plot
    # feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    # feat_imp.plot(kind='bar', title='Feature Importances')
    # plt.ylabel('Feature Importance Score')

# ----------当前最优参数----------------
def get_tuned_xgb():
    return XGBClassifier(
        # learning_rate=0.005,  # eta
        # n_estimators=2660,
        # max_depth=5,
        # min_child_weight=5,
        # gamma=0.7,
        # subsample=0.8,
        # colsample_bytree=0.8,
        # reg_alpha=0,        # alpha
        # reg_lambda=1,       # lambda
        objective='binary:logistic',
        nthread=n_thread,
        seed=36)

xgb_model = get_tuned_xgb()

# -----------只用当前参数训练一个模型-------------
# print('训练模型中...')
#
train_df=train_df[:2000]
modelfit(xgb_model, train_df, useTrainCV=True)

# -----------END 只用当前参数训练一个模型---------

# ---------------格点搜索----------------------
# 格点搜索参数
# param_test = {
#
# }
#
# gsearch = GridSearchCV(
#     estimator=xgb_model,
#     param_grid=param_test,
#     scoring='roc_auc',
#     iid=False,
#     cv=5
# )
#
# print('正在搜索...')
# gsearch.fit(train_df[predictors], train_df[target])
#
# # 输出搜索结果
# print('\n当前格点搜索的参数：\n', param_test)
# print('\ngsearch.best_params_:', gsearch.best_params_,)
# print('\ngsearch.best_score_:', gsearch.best_score_, )

# ---------------END 格点搜索----------------



# 使用模型对test集合进行预测,第一列为0的概率，第二列为1的概率
# dtest_predictions = xgb_model.predict_proba(test_df[predictors])
# pickle.dump(xgb_model, open('../models/data_3_xgb.model', 'wb'))

time_spend = time.time() - time_begin
print('\n运行时间：%d 秒，约%d分钟\n' % (time_spend, time_spend // 60))