# -*- coding=utf-8 -*-

# 用五个模型生成的stacking特征进行第二层的学习，做出预测并存储结果

print('导入算法包...')
import time
import getpass
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
import sklearn
from sklearn.model_selection import GridSearchCV
from xgboost.sklearn import XGBClassifier

time_begin = time.time()

# ---------------读取数据---------------------
DATA_DIR = '../../data/data_7/'
print('读取数据...\n位置：', DATA_DIR)

y_train = pd.read_csv(DATA_DIR + 'train_combined.csv', encoding='gbk')['target']
test_id = pd.read_csv('../../data/data_raw/test.csv', encoding='gbk')['id']

z_train1_df = pd.read_csv(DATA_DIR + 'z_train1.csv', encoding='gbk')
z_train2_df = pd.read_csv(DATA_DIR + 'z_train2.csv', encoding='gbk')
z_train3_df = pd.read_csv(DATA_DIR + 'z_train3.csv', encoding='gbk')
z_train_df = pd.concat([z_train1_df, z_train2_df, z_train3_df], axis=1)

z_test1_df = pd.read_csv(DATA_DIR + 'z_test1.csv', encoding='gbk')
z_test2_df = pd.read_csv(DATA_DIR + 'z_test2.csv', encoding='gbk')
z_test3_df = pd.read_csv(DATA_DIR + 'z_test3.csv', encoding='gbk')
z_test_df = pd.concat([z_test1_df, z_test2_df, z_test3_df], axis=1)

target = 'target'
print('数据量：', z_train_df.shape)


# --------------END 读取数据------------------

def modelfit(model, z_train, y_train, useTrainCV=True, cv_fold=5, early_stopping_rounds=50):
    if useTrainCV:
        print('交叉验证自动设定迭代次数...')
        xgb_param = model.get_xgb_params()
        xgtrain = xgb.DMatrix(z_train.values, label=y_train.values)
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
    model.fit(z_train, y_train, eval_metric='auc')
    # predict
    dtrain_predictions = model.predict(z_train)
    dtrain_predprob = model.predict_proba(z_train)[:, 1]
    # print model report
    print('\nModel Report')
    print('Accuracy:%f' % sklearn.metrics.accuracy_score(y_train.values, dtrain_predictions))
    print('AUC Score (Train): %f' % sklearn.metrics.roc_auc_score(y_train, dtrain_predprob))
    # plot
    feat_imp = pd.Series(model.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')


# ----------当前最优参数----------------
def get_tuned_xgb():
    return XGBClassifier(
        learning_rate=0.01,
        n_estimators=1000,
        max_depth=3,
        min_child_weight=5,
        gamma=0.05,
        subsample=0.7,
        colsample_bytree=1.0,
        reg_lambda=0.1,  # lambda
        reg_alpha=0.1,  # alpha
        objective='binary:logistic',
        nthread=-1,
        seed=36)


xgb_model = get_tuned_xgb()

# -----------只用当前参数训练一个模型-------------
print('训练模型中...')
#
modelfit(xgb_model, z_train=z_train_df, y_train=y_train, useTrainCV=True)

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
#     cv=5,
#     n_jobs=-1)
#
# print('正在搜索...')
# gsearch.fit(z_train_df, y_train)
#
# # 输出搜索结果
# print('\n当前格点搜索的参数：\n', param_test)
# print('\ngsearch.best_params_:', gsearch.best_params_, )
# print('\ngsearch.best_score_:', gsearch.best_score_, )

# ---------------END 格点搜索----------------


# ---------------生成预测结果--------------
# 对test集进行预测,第一列为0的概率，第二列为1的概率
# -----------生成并保存预测结果---------------
dtest_predictions = xgb_model.predict_proba(z_test_df)[:, 1]

print("存储数据中：")
print('位置:',DATA_DIR)
dtest_predictions = pd.DataFrame(dtest_predictions, columns=['predict'])
dtest_predictions = pd.DataFrame(pd.concat([test_id, dtest_predictions], axis=1))
dtest_predictions.to_csv(DATA_DIR +'submission.csv', encoding='gbk', index=False)

# -----------输出时间-----------------
time_spend = time.time() - time_begin
print('\n运行时间：%d 秒，约%d分钟\n' % (time_spend, time_spend // 60))