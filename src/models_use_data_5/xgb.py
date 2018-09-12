# -*- coding=utf-8 -*-

print('导入算法包...')
import time
import getpass
import pandas as pd
import xgboost as xgb
import sklearn
from sklearn.model_selection import GridSearchCV
from xgboost.sklearn import XGBClassifier
time_begin = time.time()


# ---------------读取数据---------------------
DATA_DIR = '../../data/data_5/'
print('读取数据...\n位置：', DATA_DIR)

train_df = pd.read_csv( DATA_DIR + 'train_engineered.csv', encoding='gbk')
predictors = train_df.columns.drop(['id','target'])
target = 'target'

if getpass.getuser()=='stone':
    train_df=train_df[:2000]

print('数据量：', train_df.shape)
# --------------END 读取数据------------------


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
            eval_metric='auc')
    # predict
    dtrain_predictions = model.predict(dtrain[predictors])
    dtrain_predprob = model.predict_proba(dtrain[predictors])[:, 1]
    # print model report
    print('\nModel Report')
    print('Accuracy:%f' % sklearn.metrics.accuracy_score(dtrain[target].values, dtrain_predictions))
    print('AUC Score (Train): %f' % sklearn.metrics.roc_auc_score(dtrain[target], dtrain_predprob))
    # plot
    feat_imp = pd.Series(model.feature_importances_, index=predictors).sort_values(ascending=False)
    # feat_imp.plot(kind='bar', title='Feature Importances')
    # plt.ylabel('Feature Importance Score')
    return feat_imp

# ----------当前最优参数----------------
def get_tuned_xgb():
    return XGBClassifier(
        learning_rate=0.1,  # eta
        n_estimators=1000,
        # max_depth=3,
        # min_child_weight=1,
        # gamma=0.1,
        # subsample=0.96,
        # colsample_bytree=0.94,
        # reg_alpha=1e-7,        # alpha
        # reg_lambda=6,       # lambda
        objective='binary:logistic',
        nthread=-1,
        seed=36)

xgb_model = get_tuned_xgb()


# -----------只用当前参数训练一个模型-------------
print('训练模型中...')

feat_imp = modelfit(xgb_model, train_df, useTrainCV=True)

# -----------END 只用当前参数训练一个模型---------


# ---------------格点搜索----------------------
# 格点搜索参数
#param_test = {
#   'n_estimators':[1000, 1200, 1300, 1400]
#}
#
#gsearch = GridSearchCV(
#    estimator=xgb_model,
#    param_grid=param_test,
#    scoring='roc_auc',
#    iid=False,
#    cv=5,
#    n_jobs=-1
#)
#
#print('正在搜索...')
#gsearch.fit(train_df[predictors], train_df[target])
#
# 输出搜索结果
#print('\n当前格点搜索的参数：\n', param_test)
#print('\ngsearch.best_params_:', gsearch.best_params_,)
#print('\ngsearch.best_score_:', gsearch.best_score_, )

# ---------------END 格点搜索----------------

print("存储变量重要性：")
print('位置：../../data/data_5/feat_imp_xgb3.csv')
feat_imp.to_csv('../../data/data_5/feat_imp_xgb3.csv', encoding='gbk')


time_spend = time.time() - time_begin
print('\n运行时间：%d 秒，约%d分钟\n' % (time_spend, time_spend // 60))
