# -*- coding=utf-8 -*-


import time
import getpass
import pandas as pd
import xgboost as xgb
import sklearn
from sklearn.model_selection import GridSearchCV, KFold
from xgboost.sklearn import XGBClassifier

time_begin = time.time()

# ---------------读取数据---------------------
DATA_DIR = '../../data/data_6/'
print('读取数据...\n位置：', DATA_DIR)

train_df = pd.read_csv(DATA_DIR + 'train_ldaed.csv', encoding='gbk')
test_df = pd.read_csv(DATA_DIR + 'test_ldaed.csv', encoding='gbk')
y_train = train_df.iloc[:, -1:]
all_features_df = train_df.iloc[:, :-1].append(test_df, ignore_index=True)

kf = KFold(n_splits=4, shuffle=True, random_state=36)
row0 = train_df[train_df['target'] == 0]
row1 = train_df[train_df['target'] == 1]

for train_index, test_index in kf.split(row0):
    row0_select = row0.iloc[test_index]

analyse_part = row0_select.append(row1)

import seaborn as sns

sns.set(style='ticks')
vars = pd.Index(['log_var1', 'log_var6', 'log_var14', 'target'])
sns.pairplot(analyse_part[vars], hue='target', vars=vars.drop('target'), palette='husl')
