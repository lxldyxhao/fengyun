# -*- coding=utf-8 -*-

print('导入算法包...')
from sklearn.feature_selection import RFECV
import time
import pandas as pd
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
import getpass

time_begin = time.time()

# ---------------读取数据---------------------
print('读取数据...')
train_df = pd.read_csv('../../data/data_4/train_preprocessed1.csv', encoding='gbk')
test_df = pd.read_csv('../../data/data_4/test_preprocessed1.csv', encoding='gbk')
all_features_df = train_df.iloc[:, :-1].append(test_df, ignore_index=True)
target = 'target'
predictors = train_df.columns.drop(['id', 'target'])
# --------------END 读取数据------------------

step = 10
if getpass.getuser() == 'stone':
    train_df = train_df[:2000]
    step = 0.1
print('数据量：', train_df.shape)

# ----------------　特征选择　------------------

xgb_estimator = XGBClassifier(
    objective='binary:logistic',
    nthread=-1,
    seed=36)

print('XGB交叉验证迭代筛选特征...')
selector = RFECV(xgb_estimator, step=step, cv=5, scoring='roc_auc')
selector.fit(train_df[predictors], train_df[target])

# 筛选特征
print('筛选最优特征...')
selected_features = selector.transform(all_features_df[predictors])
support = pd.Series(selector.support_, index=predictors)

# -------------保存筛选特征后的数据--------------

print("存储选择结果中：")
print('位置：../../data/data_4/selection_result_xgb.csv')
support.to_csv('../../data/data_4/selection_result_xgb.csv', encoding='gbk')

# ----------------输出运行时间  --------------------
time_spend = time.time() - time_begin
print('\n运行时间：%d 秒，约%d分钟\n' % (time_spend, time_spend // 60))
