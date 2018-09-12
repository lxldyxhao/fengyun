# -*- coding=utf-8 -*-

# 使用去掉离群点的原始数据进行预处理
# 进行特征工程,特征间交叉加减乘除组合
# 结果存储于../../data/data_4/train_preprocessed1.csv & test_preprocessed1.csv

import pandas as pd
import numpy as np
from sklearn import preprocessing
import jieba as jb
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import PCA
from pyltp import Segmentor
import getpass

LTP_DATA_DIR = '/home/stone/ml_data/ltp_model/ltp_data_v3.4.0/cws.model'
if getpass.getuser() == 'mc':
    LTP_DATA_DIR='/home/mc/lxl/pyltp/models/cws.model'

# 读取数据
print("读取数据中：")
print('位置：../../data/data_2/train_no_large.csv & test_no_large.csv')

test_df = pd.read_csv("../../data/data_2/test_no_large.csv", encoding='gbk')
train_df = pd.read_csv("../../data/data_2/train_no_large.csv", encoding='gbk')
y_train = train_df.iloc[:, 533:]
all_features_df = train_df.iloc[:, :-1].append(test_df, ignore_index=True)

# 获得含有空值的列的索引 columns_hasnan
null_sum = all_features_df.isnull().sum()
columns_hasnan = null_sum[null_sum != 0].index

# 处理var6:用numpy产生泊松分布随机数，进行填充，注释掉的代码用于观察填充前后的分布
print('填充var6...')
tmp = all_features_df['var6'][:]
rand = np.random.poisson(lam=30, size=tmp.isnull().sum())
tmp[np.isnan(tmp)] = rand
all_features_df['var6'] = tmp[:]
columns_hasnan = columns_hasnan.drop('var6')

# -----------填充空值--------------
print('填充空值...')
# 先填充文本数据,统一填充“空值"
text_index = [19, 20, 21, 22, 23, 24, 25, 45, 124, 125, 126]
for index, item in enumerate(text_index):
    text_index[index] = 'var' + str(item)
all_features_df[text_index] = all_features_df[text_index].fillna('空值')[:]
for i in text_index:
    if i in columns_hasnan:
        columns_hasnan = columns_hasnan.drop(i)

# 再填充分类数据：但文本已经填完，其他分类数据没有 NAN
categorical_index = [3, 11, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 45, 124, 125, 126]
for index, item in enumerate(categorical_index):
    categorical_index[index] = 'var' + str(item)

# 对空值数目超过一万的数值型数据全部填充 0
for curr_var in columns_hasnan:
    tmp = all_features_df[curr_var][:]
    if tmp.isnull().sum() > 10000:
        print("当前变量的空缺大于10000：", curr_var)
        tmp = tmp.fillna(0)
        print("填充 0")
        all_features_df[curr_var] = tmp[:]
        columns_hasnan = columns_hasnan.drop([curr_var])

# 最后对其他数值型数据统一填充中位数
for curr_var in columns_hasnan:
    print("填充变量：", curr_var)
    tmp = all_features_df[curr_var][:]
    print('中位数：', tmp.median())
    tmp = tmp.fillna(tmp.median())
    all_features_df[curr_var] = tmp[:]
    columns_hasnan = columns_hasnan.drop([curr_var])

# 唯一一个有负值的变量
all_features_df['var47'] = all_features_df['var47'].apply(lambda x: -1000 if x < -1000 else x)

# ---------处理三个可能包含通话次数的文本特征--------------

print('统计通话次数...')
curr_var = 'var124'
all_features_df[curr_var] = all_features_df[curr_var].apply(lambda str:
                                                            0 if str == '空值' else len(str.strip().split('@')))
curr_var = 'var125'
all_features_df[curr_var] = all_features_df[curr_var].apply(lambda str: str.replace(' ', ''))
all_features_df[curr_var] = all_features_df[curr_var].apply(lambda str: ' '.join(str.strip().split('@')))
all_features_df[curr_var] = all_features_df[curr_var].apply(lambda str: ' '.join(str.strip().split('；')))
all_features_df[curr_var] = all_features_df[curr_var].apply(lambda str:
                                                            0 if str == '空值' else len(str.strip().split(' ')))
curr_var = 'var126'
all_features_df[curr_var] = all_features_df[curr_var].apply(lambda str: str.replace(' ', ''))
all_features_df[curr_var] = all_features_df[curr_var].apply(lambda str: ' '.join(str.strip().split('@')))
all_features_df[curr_var] = all_features_df[curr_var].apply(lambda str: ' '.join(str.strip().split('；')))
all_features_df[curr_var] = all_features_df[curr_var].apply(lambda str:
                                                            0 if str == '空值' else len(str.strip().split(' ')))
categorical_index = pd.Index(categorical_index).drop(['var124', 'var125', 'var126'])
text_index = pd.Index(text_index).drop(['var124', 'var125', 'var126'])
all_features_df = all_features_df.rename(columns={'var124': 'var124_num',
                                                  'var125': 'var125_num',
                                                  'var126': 'var126_num'})

# ------------数值型数据标准化－－－－－－－－－－
# 数值型数据标准化,使用标准的scaler，以后可以有不同的处理方法
print('数值型数据标准化...')
numerical_columns = (all_features_df.columns).drop(categorical_index).drop('id')
numerical_features = all_features_df[numerical_columns][:]
X_scaled = preprocessing.scale(numerical_features)
X_scaled = pd.DataFrame(X_scaled, columns=numerical_columns)
all_features_df[numerical_columns] = X_scaled

# -----------数字型分类变量：使用one_hot------------
print('处理数字型分类变量...')
categorical_index = pd.Index(categorical_index)
num_categorical_index = categorical_index.drop(text_index)
num_categorical_features = all_features_df[num_categorical_index][:]
for i in num_categorical_index:
    num_categorical_features[i] = num_categorical_features[i].astype('category')
# 使用dummies 获得one_hot表示
num_categorical_dummies = pd.get_dummies(num_categorical_features, prefix=num_categorical_index)
all_features_df = all_features_df.drop(num_categorical_index, axis=1)
all_features_df = all_features_df.join(num_categorical_dummies)

# -----------------文本：tfidf------------------
print('处理文本变量:')
text_features = all_features_df[text_index][:]
all_features_df = all_features_df.drop(text_index, axis=1)

# -----------处理var19 职业--------------
print('处理var19...')
# 替换var19的同义词
text_features['var19'][text_features['var19'] == '后厨'] = '厨师'
text_features['var19'][text_features['var19'] == '厨房'] = '厨师'
text_features['var19'][text_features['var19'] == '公司受雇店长'] = '公司受雇员工'
text_features['var19'][text_features['var19'] == '负责人'] = '公司受雇员工'
text_features['var19'][text_features['var19'] == '副店长'] = '公司受雇员工'
text_features['var19'][text_features['var19'] == '部长'] = '公司受雇员工'
text_features['var19'][text_features['var19'] == '公司员工'] = '公司受雇员工'
text_features['var19'][text_features['var19'] == '一般员工'] = '公司受雇员工'
text_features['var19'][text_features['var19'] == '普通员工'] = '公司受雇员工'
text_features['var19'][text_features['var19'] == '受雇员工'] = '公司受雇员工'
text_features['var19'][text_features['var19'] == '品质主管'] = '公司受雇员工'
text_features['var19'][text_features['var19'] == '生产主管'] = '公司受雇员工'
text_features['var19'][text_features['var19'] == '前台'] = '公司受雇员工'
text_features['var19'][text_features['var19'] == '司机'] = '公司受雇员工'
text_features['var19'][text_features['var19'] == '设计师'] = '公司受雇员工'
text_features['var19'][text_features['var19'] == '拓展部经理'] = '公司受雇员工'
text_features['var19'][text_features['var19'] == '经理'] = '公司受雇员工'
text_features['var19'][text_features['var19'] == '快递'] = '公司受雇员工'
text_features['var19'][text_features['var19'] == '快递员'] = '公司受雇员工'
text_features['var19'][text_features['var19'] == '高管'] = '公司受雇员工'
text_features['var19'][text_features['var19'] == '销售管理'] = '公司受雇员工'
text_features['var19'][text_features['var19'] == '销售部长'] = '公司受雇员工'
text_features['var19'][text_features['var19'] == '销售管理'] = '公司受雇员工'
text_features['var19'][text_features['var19'] == '员工'] = '公司受雇员工'
text_features['var19'][text_features['var19'] == '技工'] = '公司受雇员工'
text_features['var19'][text_features['var19'] == '职工'] = '私企公司职工'
text_features['var19'][text_features['var19'] == '单位职工'] = '私企公司职工'
text_features['var19'][text_features['var19'] == '民营事业单位职工'] = '私企公司职工'
text_features['var19'][text_features['var19'] == '失业'] = '自由职业者'
text_features['var19'][text_features['var19'] == '失业人员'] = '自由职业者'
text_features['var19'][text_features['var19'] == '家庭主妇'] = '自由职业者'
text_features['var19'][text_features['var19'] == '创业人员'] = '自由职业者'
text_features['var19'][text_features['var19'] == '自雇创业人员'] = '自主创业人员'

# 用jieba分词，这里用了全模式，可能出现重复的字
text_features['var19'] = text_features['var19'].apply(lambda str: str.strip())
cuted_feature_var19 = text_features['var19'].apply(lambda str: ' '.join(jb.cut(str, cut_all=True)))
# 统计词频
count_vect = CountVectorizer()
count_var19 = count_vect.fit_transform(cuted_feature_var19.values)
# 转换为tfidf
tfidf_transformer = TfidfTransformer()
tfidf_var19 = tfidf_transformer.fit_transform(count_var19)
col_var19 = ['var19_' + str(i) for i in range(1, 36, 1)]
# 存入特征dataframe中
tfidf_var19_pd = pd.DataFrame(tfidf_var19.toarray(), columns=col_var19)
all_features_df = all_features_df.join(tfidf_var19_pd)
text_features = text_features.drop('var19', axis=1)

# --------------地址---------------
# --------------var22 --------------
seg = Segmentor()
seg.load(LTP_DATA_DIR)

print('处理var22...')
# 去掉空格,分词,添加到text_feature中
text_features['var22'] = text_features['var22'].apply(lambda str: ''.join(str.strip().split(' ')))
text_features['var22'] = text_features['var22'].apply(lambda str: ''.join(str.split('"')))
text_features['var22'][text_features['var22'] == '请选择开户地'] = '空值'
cuted_var22 = text_features['var22'].apply(lambda str: pd.Series([i for i in seg.segment(str)]))
cuted_var22.set_axis(axis=1, labels=['var22_sheng', 'var22_1', 'var22_2', 'var22_3', 'var22_4', 'var22_5'])
cuted_var22 = cuted_var22.fillna('空值')
text_features = text_features.drop('var22', axis=1).join(cuted_var22)

# -----------var23分词--------------
print('处理var23...')
# 去掉空格,分词,添加到text_feature中
curr_var = 'var23'
text_features[curr_var] = text_features[curr_var].apply(lambda str: ''.join(str.strip().split(' ')))
cuted_var23 = text_features[curr_var].apply(lambda str: pd.Series([i for i in seg.segment(str)]))
cuted_var23.set_axis(axis=1, labels=['var23_sheng', 'var23_1', 'var23_2', 'var23_3',
                                     'var23_4', 'var23_5', 'var23_6', 'var23_7', 'var23_8', 'var23_9'])
cuted_var23 = cuted_var23.fillna('空值')
text_features = text_features.drop('var23', axis=1).join(cuted_var23)

# 对所有的地址词，只保留前两个字
for col in text_features.columns:
    text_features[col] = text_features[col].apply(lambda str: str[:2])

# --------对基本信息中的省,用tfidf编码-----------
print('处理所有的省...')
sheng_features = text_features['var20'] + ' ' + text_features['var22_sheng'] + ' ' + \
                 text_features['var23_sheng'] + ' ' + text_features['var24']
# 统计词频
count_vect = CountVectorizer()
count_var_sheng = count_vect.fit_transform(sheng_features.values)
# 转换为tfidf
tfidf_transformer = TfidfTransformer()
tfidf_var_sheng = tfidf_transformer.fit_transform(count_var_sheng)
col_var_sheng = ['var_sheng_' + str(i) for i in range(0, tfidf_var_sheng.shape[1], 1)]
# 存入特征dataframe中
tfidf_var_sheng = pd.DataFrame(tfidf_var_sheng.toarray(), columns=col_var_sheng)
all_features_df = all_features_df.join(tfidf_var_sheng)
text_features = text_features.drop(['var20', 'var22_sheng', 'var23_sheng', 'var24'], axis=1)

# --------对基本信息中的市,用tfidf编码-----------
print('处理所有的市及以下文本...')
shi_features = text_features.apply(lambda s: s + ' ').T.sum()
# 统计词频
count_vect = CountVectorizer()
count_var_shi = count_vect.fit_transform(shi_features.values)
# 转换为tfidf
tfidf_transformer = TfidfTransformer()
tfidf_var_shi = tfidf_transformer.fit_transform(count_var_shi)
col_var_shi = ['var_shi_' + str(i) for i in range(0, tfidf_var_shi.shape[1], 1)]
# 存入特征dataframe中
tfidf_var_shi = pd.DataFrame(tfidf_var_shi.toarray(), columns=col_var_shi)
# PCA降维
print('PCA降维...')
pca = PCA(n_components=200,
          svd_solver='randomized',
          random_state=36)
pca.fit(tfidf_var_shi)
shi_features = pd.DataFrame(pca.transform(tfidf_var_shi),
                            columns=['var_shi_'+str(i) for i in range(0,200,1)])
all_features_df = all_features_df.join(shi_features)

#  --------------- END 文本：tfidf------------------

# ---------保存预处理后的数据----------------
print("存储数据中：")
print('位置：../../data/data_4/train_preprocessed1.csv & test_preprocessed1.csv')
train_feat = all_features_df[:24309]
train_df = train_feat.join(y_train)

test_feat = all_features_df[24309:]
test_df = test_feat

train_df.to_csv('../../data/data_4/train_preprocessed1.csv', encoding='gbk', index=False)
test_df.to_csv('../../data/data_4/test_preprocessed1.csv', encoding='gbk', index=False)
