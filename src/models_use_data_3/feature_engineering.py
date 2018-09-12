# -*- coding=utf-8 -*-

# 使用去掉离群点的原始数据进行预处理
# 填充空值，处理分类变量/文本变量，便于用于不能接受空值的模型
# 结果存储于../../data/data_3/train_preprocessed1.csv & test_preprocessed1.csv

import pandas as pd
import numpy as np
from sklearn import preprocessing
import jieba as jb
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

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
# fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))
# axis1.set_title('Original values  ')
# axis2.set_title('New values ')
# tmp.hist(bins=70,ax=axis1)
rand = np.random.poisson(lam=30, size=tmp.isnull().sum())
tmp[np.isnan(tmp)] = rand
# tmp.hist(bins=70,ax=axis2)
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
    if tmp.isnull().sum() == all_features_df.shape[0]:
        all_features_df = all_features_df.drop(curr_var, axis=1)
        columns_hasnan = columns_hasnan.drop([curr_var])
        continue
    if tmp.isnull().sum() > 10000:
        print("当前变量的空缺大于10000：", curr_var)
        if curr_var == 'var135':
            all_features_df = all_features_df.drop(curr_var, axis=1)
            columns_hasnan = columns_hasnan.drop([curr_var])
            continue
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
# 考虑测试集合中var47和其他变量的相关关系？借以推断测试数据集中的０的原值
all_features_df['var47'][all_features_df['var47'] < -1000] = -1000



def sum_columns(start,end, all_features_df):
    sum_list = []
    for i in range(start, end, 1):
        curr_var = 'var' + str(i)
        sum_list.append(curr_var)
    sum_feat = pd.DataFrame(all_features_df[sum_list].T.sum())
    sum_feat.columns = ['var' + str(start) + '~' + str(end)]
    all_features_df = all_features_df.join(sum_feat)
    if(start==489):
        all_features_df = all_features_df.drop(sum_list,axis=1)
    return all_features_df


all_features_df = sum_columns(136,152,all_features_df)
all_features_df = sum_columns(152,166,all_features_df)
all_features_df = sum_columns(168,187,all_features_df)
all_features_df = sum_columns(189,209,all_features_df)
all_features_df = sum_columns(211,228,all_features_df)
all_features_df = sum_columns(230,236,all_features_df)
all_features_df = sum_columns(237,266,all_features_df)
all_features_df = sum_columns(268,304,all_features_df)
all_features_df = sum_columns(307,326,all_features_df)
all_features_df = sum_columns(328,347,all_features_df)
all_features_df = sum_columns(349,369,all_features_df)
all_features_df = sum_columns(371,389,all_features_df)
all_features_df = sum_columns(391,428,all_features_df)
all_features_df = sum_columns(430,467,all_features_df)
all_features_df = sum_columns(469,489,all_features_df)
all_features_df = sum_columns(489,533,all_features_df)




# ---------处理三个可能包含通话次数的文本特征--------------
print('统计通话次数...')
curr_var = 'var124'
all_features_df[curr_var] = all_features_df[curr_var].apply(lambda str:
                                                            0 if str=='空值' else len(str.strip().split('@')))
curr_var = 'var125'
all_features_df[curr_var] = all_features_df[curr_var].apply(lambda str: str.replace(' ',''))
all_features_df[curr_var] = all_features_df[curr_var].apply(lambda str:' '.join(str.strip().split('@')))
all_features_df[curr_var] = all_features_df[curr_var].apply(lambda str:' '.join(str.strip().split('；')))
all_features_df[curr_var] = all_features_df[curr_var].apply(lambda str:
                                                            0 if str=='空值' else len(str.strip().split(' ')))
curr_var = 'var126'
all_features_df[curr_var] = all_features_df[curr_var].apply(lambda str: str.replace(' ',''))
all_features_df[curr_var] = all_features_df[curr_var].apply(lambda str:' '.join(str.strip().split('@')))
all_features_df[curr_var] = all_features_df[curr_var].apply(lambda str:' '.join(str.strip().split('；')))
all_features_df[curr_var] = all_features_df[curr_var].apply(lambda str:
                                                            0 if str=='空值' else len(str.strip().split(' ')))
categorical_index = pd.Index(categorical_index).drop(['var124','var125','var126'])
text_index = pd.Index(text_index).drop(['var124','var125','var126'])
all_features_df = all_features_df.rename(columns={'var124':'var124_num',
                                                  'var125':'var125_num',
                                                  'var126':'var126_num'})

# ------------数值型数据标准化－－－－－－－－－－
# 数值型数据标准化,使用标准的scaler，以后可以有不同的处理方法
print('数值型数据标准化...')
numerical_columns = (all_features_df.columns).drop(categorical_index)
numerical_features = all_features_df[numerical_columns][:]
X_scaled = preprocessing.scale(numerical_features)
X_scaled = pd.DataFrame(X_scaled,columns=numerical_columns)
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


# -----------文本类型的分类变量：tfidf--------------
print('处理文本变量:')
text_features = all_features_df[text_index][:]

# -----------处理var19--------------
print('处理var19...')
# 替换var19的同义词
text_features['var19'][text_features['var19']=='后厨'] = '厨师'
text_features['var19'][text_features['var19']=='厨房'] = '厨师'
text_features['var19'][text_features['var19']=='公司受雇店长'] = '公司受雇员工'
text_features['var19'][text_features['var19']=='负责人'] = '公司受雇员工'
text_features['var19'][text_features['var19']=='副店长'] = '公司受雇员工'
text_features['var19'][text_features['var19']=='部长'] = '公司受雇员工'
text_features['var19'][text_features['var19']=='公司员工'] = '公司受雇员工'
text_features['var19'][text_features['var19']=='一般员工'] = '公司受雇员工'
text_features['var19'][text_features['var19']=='普通员工'] = '公司受雇员工'
text_features['var19'][text_features['var19']=='受雇员工'] = '公司受雇员工'
text_features['var19'][text_features['var19']=='品质主管'] = '公司受雇员工'
text_features['var19'][text_features['var19']=='生产主管'] = '公司受雇员工'
text_features['var19'][text_features['var19']=='前台'] = '公司受雇员工'
text_features['var19'][text_features['var19']=='司机'] = '公司受雇员工'
text_features['var19'][text_features['var19']=='设计师'] = '公司受雇员工'
text_features['var19'][text_features['var19']=='拓展部经理'] = '公司受雇员工'
text_features['var19'][text_features['var19']=='经理'] = '公司受雇员工'
text_features['var19'][text_features['var19']=='快递'] = '公司受雇员工'
text_features['var19'][text_features['var19']=='快递员'] = '公司受雇员工'
text_features['var19'][text_features['var19']=='高管'] = '公司受雇员工'
text_features['var19'][text_features['var19']=='销售管理'] = '公司受雇员工'
text_features['var19'][text_features['var19']=='销售部长'] = '公司受雇员工'
text_features['var19'][text_features['var19']=='销售管理'] = '公司受雇员工'
text_features['var19'][text_features['var19']=='员工'] = '公司受雇员工'
text_features['var19'][text_features['var19']=='技工'] = '公司受雇员工'
text_features['var19'][text_features['var19']=='职工'] = '私企公司职工'
text_features['var19'][text_features['var19']=='单位职工'] = '私企公司职工'
text_features['var19'][text_features['var19']=='民营事业单位职工'] = '私企公司职工'
text_features['var19'][text_features['var19']=='失业'] = '自由职业者'
text_features['var19'][text_features['var19']=='失业人员'] = '自由职业者'
text_features['var19'][text_features['var19']=='家庭主妇'] = '自由职业者'
text_features['var19'][text_features['var19']=='创业人员'] = '自由职业者'
text_features['var19'][text_features['var19']=='自雇创业人员'] = '自主创业人员'

# 用jieba分词，这里用了全模式，可能出现重复的字
text_features['var19'] = text_features['var19'].apply(lambda str:str.strip())
cuted_feature_var19 = text_features['var19'].apply(lambda str:' '.join(jb.cut(str,cut_all=True)))
# 统计词频
count_vect = CountVectorizer()
count_var19  = count_vect.fit_transform(cuted_feature_var19.__array__())
# 转换为tfidf
tfidf_transformer = TfidfTransformer()
tfidf_var19 = tfidf_transformer.fit_transform(count_var19)
col_var19 = ['var19_' + i for i in list(count_vect.vocabulary_.keys())]
# 存入特征dataframe中
tfidf_var19_pd = pd.DataFrame(tfidf_var19.toarray(), columns=col_var19)
all_features_df = all_features_df.join(tfidf_var19_pd)
all_features_df = all_features_df.drop('var19', axis=1)

# -----------处理var20--------------
print('处理var20...')
# 只保留省的前两个字
tmp = all_features_df['var20'][:]
tmp = tmp.apply(lambda str:str[:2])
all_features_df['var20'] = tmp.astype('category')
# one_hot处理
var20_dummies = pd.get_dummies(all_features_df['var20'], prefix='var20')
all_features_df = all_features_df.drop('var20', axis=1)
all_features_df = all_features_df.join(var20_dummies)

# -----------处理var22--------------
print('处理var22...')
# 去掉空格
text_features['var22'] = text_features['var22'].apply(lambda str:''.join(str.strip().split(' ')))
text_features['var22'] = text_features['var22'].apply(lambda str:''.join(str.split('"')))
text_features['var22'][text_features['var22']=='请选择开户地']='空值'
# 分词
cuted_var22 = text_features['var22'].apply(lambda str:pd.Series([i for i in jb.cut(str)]))
cuted_var22[2][cuted_var22[2]=='市']=np.nan
cuted_var22[2][cuted_var22[2]=='区']=np.nan

# 分开的四列大概起个名字：省／市／民族１／民族２
cuted_var22.set_axis(axis=1,labels=['var22_sheng','var22_shi','var22_zu1','var22_zu2'])
cuted_var22 = cuted_var22.fillna('空值')

# 手工处理特殊值
cuted_var22['var22_sheng'][cuted_var22['var22_zu1'] == '阳'] = '湖南'
cuted_var22['var22_shi'][cuted_var22['var22_zu1'] == '阳'] = '益阳'
cuted_var22['var22_zu1'][cuted_var22['var22_zu1'] == '阳'] = '空值'
cuted_var22['var22_shi'][cuted_var22['var22_zu1']=='沙市'] = '三沙市'
cuted_var22['var22_zu1'][cuted_var22['var22_zu1']=='沙市'] = '空值'

# 对所有的词，只保留前两个字
for col in cuted_var22.columns:
    cuted_var22[col] = cuted_var22[col].apply(lambda str:str[:2])
    if col!='var22_shi':
        cuted_var22[col] = cuted_var22[col].astype('category')

# 把 var22_市 加入到特征中
all_features_df = all_features_df.join(cuted_var22['var22_shi'])
# 进行one_hot编码，不展开市的一列
var22_dummies = pd.get_dummies(cuted_var22.drop('var22_shi',axis=1),
                               prefix=list(cuted_var22.columns.drop('var22_shi')))
all_features_df = all_features_df.drop('var22', axis=1)
all_features_df = all_features_df.join(var22_dummies)

# -----------处理var23--------------
print('处理var23...')
curr_var = 'var23'
all_features_df[curr_var] = all_features_df[curr_var].apply(lambda str:''.join(str.strip().split(' ')))
all_features_df[curr_var+'_sheng'] = all_features_df[curr_var].apply(lambda str:str[:2]).astype('category')
# one_hot处理
curr_dummies = pd.get_dummies(all_features_df[curr_var+'_sheng'], prefix=curr_var+'_sheng')
all_features_df = all_features_df.drop(curr_var+'_sheng', axis=1)
all_features_df = all_features_df.join(curr_dummies)

# -----------处理var24--------------
print('处理var24...')
curr_var = 'var24'
all_features_df[curr_var] = all_features_df[curr_var].apply(lambda str:''.join(str.strip().split(' ')))
all_features_df[curr_var] = all_features_df[curr_var].apply(lambda str:str[:2]).astype('category')
# one_hot处理
curr_dummies = pd.get_dummies(all_features_df[curr_var], prefix=curr_var)
all_features_df = all_features_df.drop(curr_var, axis=1)
all_features_df = all_features_df.join(curr_dummies)

print("丢弃地级市和空值列...")
all_features_df = all_features_df.drop(['var21','var23','var25','var45','var22_shi'], axis=1)
# drop所有表示空值的列
for i in all_features_df.columns:
    if '空值' in i:
        all_features_df = all_features_df.drop(i,axis=1)

# ---------保存预处理后的数据----------------
# 处理过的特征再分开成训练集和测试集并存储
print("存储数据中：")
train_feat = all_features_df[:24309]
train_df = train_feat.join(y_train)

test_feat = all_features_df[24309:]
test_df = test_feat

train_df.to_csv('../../data/data_3/train_engineered.csv',encoding='gbk',index=False)
test_df.to_csv('../../data/data_3/test_engineered.csv',encoding='gbk',index=False)


