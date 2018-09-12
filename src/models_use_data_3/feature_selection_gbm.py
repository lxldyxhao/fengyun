# -*- coding=utf-8 -*-

print('导入算法包...')
from sklearn.feature_selection import RFECV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
import time
import pandas as pd
import getpass
from sklearn.model_selection import GridSearchCV

time_begin = time.time()

# ---------------读取数据---------------------
print('读取数据...')
train_df = pd.read_csv('../../data/data_3/train_preprocessed1.csv', encoding='gbk')
test_df = pd.read_csv('../../data/data_3/test_preprocessed1.csv', encoding='gbk')
all_features_df = train_df.iloc[:, :-1].append(test_df, ignore_index=True)
target = 'target'
predictors = train_df.columns.drop(['id', 'target'])
# --------------END 读取数据------------------

step = 10
if getpass.getuser() == 'stone':
    train_df = train_df[:200]
    step = 100
print('数据量：', train_df.shape)

# ----------------　特征选择　------------------

estimator = GradientBoostingClassifier(
    learning_rate=0.015,
    n_estimators=1600,
    max_depth=5,
    min_samples_split=550,
    min_samples_leaf=48,
    max_features=27,
    subsample=0.8,
    random_state=36)





print('GBM 交叉验证迭代筛选特征...')
selector = new_RFECV(estimator, step=step, cv=5, scoring='roc_auc', n_jobs=3)
selector.fit(train_df[predictors], train_df[target])


support = selector.selected_features

# -------------保存筛选特征后的数据--------------

# print("存储选择结果中：")
print('位置：../../data/data_3/selection_result_gbm.csv')
support.to_csv('../../data/data_3/selection_result_gbm.csv', encoding='gbk')

# ----------------输出运行时间  --------------------
time_spend = time.time() - time_begin
print('\n运行时间：%d 秒，约%d分钟\n' % (time_spend, time_spend // 60))
