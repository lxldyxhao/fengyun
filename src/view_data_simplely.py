# -*- encoding = utf-8 -*-
# 本段代码的作用主要是了解数据的面貌，了解相关性，没做过多处理。
# 存储了上述数据在../data/data_1/train.csv & test.csv
# 为了尽快跑一个xgboost模型，用简单方法去掉了文本和空值超过半数的变量，并将
# 处理结果存放在：../data/data_1/effective_notxt_train.csv & effective_notxt_test.csv



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

test_df = pd.read_csv("../data/data_raw/test.csv",encoding='gbk')
train_df = pd.read_csv("../data/data_raw/train.csv",encoding='gbk')

x_train = train_df.iloc[:,1:533]
y_train = train_df.iloc[:,533:]


all_features_df = train_df.iloc[:,:-1].append(test_df,ignore_index=True)

# 不同类型的特征
basic_features_df = all_features_df.iloc[:,1:26]
call_features_df = all_features_df.iloc[:,26:134]
credit_features_df = all_features_df.iloc[:,134:]

# 产生分类变量索引
categorical_index = [3,11,16,17,18,19,20,21,22,23,24,25,45,124,125,126]
for index,item in enumerate(categorical_index):
    categorical_index[index] = 'var'+str(item)

# 可能是资产总数,相关性比较高
# 只有1000,2000,3000...5000的值，但不是分类变量,可能是某种额度信息
# var3-var5可能都是分类变量

# 画数据的分布图
def distribution_plot(i):
    var_test = basic_features_df['var'+ i ][:]
    var_test.sort()
    plt.plot(var_test.values,'bo')


# 去掉一半以上为空的特征
# null_sum = all_fetures_df.isnull().sum(axis=0)
# effective_features_df = all_fetures_df.drop(list(null_sum[null_sum>15000].index),axis=1)


# 去掉文本features
# text_columns = ['var19','var20','var21','var22','var23','var24','var25','var45','var124','var126','var125']
# effective_notxt_df = effective_features_df.drop(text_columns,axis=1)

# 将分类变量设置为categoricals，增加var_cat特征，去掉原特征
# for i in categorical_index:
#     try:
#         all_features_df[i+'_cat'] = all_features_df[i].astype('category')
#     except:
#         print(i,"droped")
# all_features_df = all_features_df.drop(['var3','var11','var16','var17','var18'],axis=1)

# 处理过的特征再分开成训练集和测试集
train_feat = all_features_df[:24309]
test_feat = all_features_df[24309:]
train_df = train_feat.join(y_train)

train_df.to_csv('../data/data_1/train.csv',encoding='gbk',index=False)
test_df.to_csv('../data/data_1/test.csv',encoding='gbk',index=False)

# heatmap graph 表示数据相关性
def heatmap(data_test):
    corr = data_test.corr()
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    f, ax = plt.subplots(figsize=(11,9))
    sns.set(style='white')
    cmap = sns.diverging_palette(220,10,as_cmap=True)
    sns.heatmap(corr,mask=mask, cmap=cmap, vmax=.3, center=0,square=True, linewidths=.5,cbar_kws={'shrink':.5})

