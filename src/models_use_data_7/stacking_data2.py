# -*- coding=utf-8 -*-

# 用调参后的模型生成用于第二层的stacking特征

import best_models as bm
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import getpass
from sklearn.metrics import roc_auc_score, mean_absolute_error, auc
import xgboost as xgb
import time

time_begin = time.time()
SEED = 36

DATA_DIR = '../../data/data_7/'
print('读取数据...\n位置：', DATA_DIR)

train_df = pd.read_csv(DATA_DIR + 'train_combined.csv', encoding='gbk')
test_df = pd.read_csv(DATA_DIR + 'test_combined.csv', encoding='gbk')
predictors = train_df.columns.drop(['id', 'target'])
target = 'target'

# if getpass.getuser() == 'stone':
#     train_df = train_df[:20]

ntrain = train_df.shape[0]
ntest = test_df.shape[0]

x_train = np.array(train_df[predictors])
y_train = train_df[target].ravel()
x_test = np.array(test_df[predictors])

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)


def get_oof(model, model_name):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((5, ntest))

    for i, (train_index, test_index) in enumerate(kf.split(x_train)):
        kf_x_train = x_train[train_index]
        kf_y_train = y_train[train_index]
        kf_x_test = x_train[test_index]

        print(model_name, 'trainning...　数据量:{},{}'.format(kf_x_train.shape, kf_y_train.shape))
        model.fit(kf_x_train, kf_y_train)

        oof_train[test_index] = model.predict_proba(kf_x_test)[:, 1]
        oof_test_skf[i, :] = model.predict_proba(x_test)[:, 1]

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


# 初始化各个调过参数的模型
# xgb_model = bm.get_tuned_xgb()
# gbm_model = bm.get_tuned_gbm()
# rf_model = bm.get_tuned_rf()
sgd_lr_model = bm.get_tuned_lr()
sgd_svm_model = bm.get_tuned_svc()

# 产生在训练集上交叉预测的列，以及在测试集上预测的平均值
# xgb_oof_train, xgb_oof_test = get_oof(xgb_model, 'XGB')
# gbm_oof_train, gbm_oof_test = get_oof(gbm_model, 'GBM')
# rf_oof_train, rf_oof_test = get_oof(rf_model, 'RF')
lr_oof_train, lr_oof_test = get_oof(sgd_lr_model, 'LR')
svm_oof_train, svm_oof_test = get_oof(sgd_svm_model, 'SVM')

# 输出训练集上交叉验证的相关指标
# print('\nXGB-CV mean_absolute_error: {}'.format(mean_absolute_error(y_train, xgb_oof_train)))
# print('XGB-CV roc_auc_score: {}'.format(roc_auc_score(y_train, xgb_oof_train)))

# print('\nGBM-CV mean_absolute_error: {}'.format(mean_absolute_error(y_train, gbm_oof_train)))
# print('GBM-CV roc_auc_score: {}'.format(roc_auc_score(y_train, gbm_oof_train)))

# print('\nRF-CV mean_absolute_error: {}'.format(mean_absolute_error(y_train, rf_oof_train)))
# print('RF-CV roc_auc_score: {}'.format(roc_auc_score(y_train, rf_oof_train)))

print('\nLR-CV mean_absolute_error: {}'.format(mean_absolute_error(y_train, lr_oof_train)))
print('LR-CV roc_auc_score: {}'.format(roc_auc_score(y_train, lr_oof_train)))

print('\nSVM-CV mean_absolute_error: {}'.format(mean_absolute_error(y_train, svm_oof_train)))
print('SVM-CV roc_auc_score: {}'.format(roc_auc_score(y_train, svm_oof_train)))

# 产生新的训练集和测试集，即各个算法在训练集上交叉预测的列的并排
z_train = np.concatenate((lr_oof_train,
                           svm_oof_train,), axis=1)
z_test = np.concatenate((lr_oof_test,
                           svm_oof_test,), axis=1)
#z_train = lr_oof_train
#z_test = lr_oof_test

print("\nz_train:{}, z_test:{}".format(z_train.shape, z_test.shape))

# 保存新的训练集和测试集
print("\n存储数据中：")
print('位置：', DATA_DIR)
z_train_pd = pd.DataFrame(z_train, columns=['LR','SVM'])
z_test_pd = pd.DataFrame(z_test, columns=['LR','SVM'])
z_train_pd.to_csv(DATA_DIR + 'z_train2.csv', encoding='gbk', index=False)
z_test_pd.to_csv(DATA_DIR + 'z_test2.csv', encoding='gbk', index=False)

# ------------输出运行时间　不需要改---------------
time_spend = time.time() - time_begin
print('\n运行时间：%d 秒，约%d分钟\n' % (time_spend, time_spend // 60))
