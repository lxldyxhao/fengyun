# -*- coding=utf-8 -*-

# 用调参后的模型生成用于第二层的stacking特征

import best_models as bm
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import getpass
from sklearn.metrics import roc_auc_score
import time
time_begin = time.time()
SEED=36

# ===============data==================
DATA_DIR = '../../data/data_4/'
print('读取数据...')
print('位置：', DATA_DIR)
train_df = pd.read_csv(DATA_DIR + 'train_preprocessed1.csv', encoding='gbk')
test_df = pd.read_csv(DATA_DIR + 'test_preprocessed1.csv', encoding='gbk')
target = 'target'
predictors = train_df.columns.drop(['id','target'])

if getpass.getuser() == 'stone':
    train_df = train_df[:200]

x_train = np.array(train_df[predictors])
y_train = train_df[target].ravel()
x_test = np.array(test_df[predictors])

# ==============END data==================

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

def get_oof(model, x_train, y_train, x_test,  model_name):
    oof_train = np.zeros((x_train.shape[0],))
    oof_test = np.zeros((x_test.shape[0],))
    oof_test_skf = np.empty((5, x_test.shape[0]))

    for i,(train_index, test_index) in enumerate(kf.split(x_train)):
        kf_x_train = x_train[train_index]
        kf_y_train = y_train[train_index]
        kf_x_test = x_train[test_index]

        print(model_name, 'trainning...　数据量:{},{}'.format(kf_x_train.shape, kf_y_train.shape))
        model.fit(kf_x_train, kf_y_train)

        oof_train[test_index] = model.predict_proba(kf_x_test)[:,1]
        oof_test_skf[i,:] = model.predict_proba(x_test)[:,1]

    oof_test[:] = oof_test_skf.mean(axis=0)
    oof_train = oof_train.reshape(-1, 1)
    oof_test = oof_test.reshape(-1, 1)

    print('{}-CV roc_auc_score: {}'.format(model_name, roc_auc_score(y_train, oof_train)))
    return oof_train, oof_test

# 初始化各个调过参数的模型,获取对应数据
svc_model = bm.get_tuned_svc()

# 产生在训练集上交叉预测的列，以及在测试集上预测的平均值
svc_oof_train, svc_oof_test = get_oof(svc_model,
                                      x_train=x_train,
                                      y_train=y_train,
                                      x_test=x_test,
                                      model_name='svc')


# 产生新的训练集和测试集，即各个算法在训练集上交叉预测的列的并排
z_train = svc_oof_train
z_test = svc_oof_test

print("\nz_train:{}, z_test:{}".format(z_train.shape, z_test.shape))

# 保存新的训练集和测试集
print("\n存储数据中：")
print('位置:', DATA_DIR)
z_train_pd = pd.DataFrame(z_train, columns=['svc'])
z_test_pd = pd.DataFrame(z_test, columns=['svc'])
z_train_pd.to_csv(DATA_DIR + 'z_train3.csv',encoding='gbk',index=False)
z_test_pd.to_csv(DATA_DIR + 'z_test3.csv',encoding='gbk',index=False)

# ------------输出运行时间　不需要改---------------
time_spend = time.time() - time_begin
print('\n运行时间：%d 秒，约%d分钟\n' % (time_spend, time_spend // 60))



