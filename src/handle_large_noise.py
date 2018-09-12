# -*- encoding = utf-8 -*-

# 处理过大的数据 outlier (处理空值之前处理，便于观察分布)
# 选择一个整数值(n*10000)，使大于这个值的数据点为不超过10个，将大于的值全部设置为这个值
# 输入：../data/data_raw/train.csv & test.csv
# 输出：../data/data_2/train_no_large.csv & test_no_large.csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据
test_df = pd.read_csv("../data/data_raw/test.csv",encoding='gbk')
train_df = pd.read_csv("../data/data_raw/train.csv",encoding='gbk')

all_features_df = train_df.iloc[:,:-1].append(test_df,ignore_index=True)
y_train = train_df.iloc[:,533:]

max = all_features_df.max()

# 存储要替换噪声的最大值
max_wan = max//10000
max_wan = max_wan.drop('id')

# 产生分类变量索引
categorical_index = [3,11,16,17,18,19,20,21,22,23,24,25,45,124,125,126]
for index,item in enumerate(categorical_index):
    categorical_index[index] = 'var'+str(item)

# 去掉所有的分类变量
max_wan_index_set = set(max_wan.index)
for c in categorical_index:
    if c in max_wan_index_set:
        max_wan = max_wan.drop(c)

# 去掉所有原本没有超大数据的列
max_wan = max_wan.drop(max_wan[max_wan==0].index)


max_wan_index_list = list(max_wan.index)
tmp_all_features = all_features_df[:]

# 逐列求得这个列的适合的最大值，使最大值的数量不超过10个
for i in max_wan_index_list:
    k = 1
    while (tmp_all_features[i] > (k*10000)).sum() > 10 and k <= max_wan[i]:
        k = k + 1
    max_wan[i] = k
    print('最大值选择 ',i,' ',max_wan[i]*10000)

# 将大于这列的最大值的数据，全部用这个最大值替换
for i in max_wan_index_list:
    if max_wan[i]==1:
        continue
    tmp = tmp_all_features[i][:]
    max = max_wan[i]*10000
    tmp[tmp > max] = max
    tmp_all_features[i] = tmp[:]


all_features_df = tmp_all_features

# 处理过的特征再分开成训练集和测试集并存储
train_feat = all_features_df[:24309][:]
train_df = train_feat.join(y_train)

test_feat = all_features_df[24309:][:]
test_df = test_feat[:]

train_df.to_csv('../data/data_2/train_no_large.csv',encoding='gbk',index=False)
test_df.to_csv('../data/data_2/test_no_large.csv',encoding='gbk',index=False)
