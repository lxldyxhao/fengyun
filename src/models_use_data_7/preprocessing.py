# encoding: utf-8

import pandas as pd
from sklearn import preprocessing
import sklearn
from math import log
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from pyltp import Segmentor
import getpass

LTP_DATA_DIR = '/home/stone/ml_data/ltp_model/ltp_data_v3.4.0/cws.model'
if getpass.getuser() == 'mc':
    LTP_DATA_DIR='/home/mc/lxl/pyltp/models/cws.model'

print('处理数据...')
test_df = pd.read_csv("../../data/data_2/test_no_large.csv", encoding='gbk')
train_df = pd.read_csv("../../data/data_2/train_no_large.csv", encoding='gbk')
y_train = train_df.iloc[:, -1:]
all_features_df = train_df.iloc[:, :-1].append(test_df, ignore_index=True)

# 离群点
all_features_df['var47'] = all_features_df['var47'].apply(lambda x: -500 if x < -500 else x)

# ============基本信息=================
basic_columns = ['var' + str(i) for i in range(1, 19, 1)]
new_var1 = all_features_df['var1'].apply(lambda x: log(x))
new_var1 = preprocessing.scale(new_var1)
all_features_df = all_features_df.join(pd.DataFrame({'log_var1': new_var1}))

new_var6 = all_features_df['var6'].fillna(all_features_df['var6'].median()) \
    .apply(lambda x: log(x + 1))
new_var6 = preprocessing.scale(new_var6)
all_features_df = all_features_df.join(pd.DataFrame({'log_var6': new_var6}))

new_var14 = all_features_df['var14'].apply(lambda x: log(x))
new_var14 = preprocessing.scale(new_var14)
all_features_df = all_features_df.join(pd.DataFrame({'log_var14': new_var14}))

basic_features = all_features_df[basic_columns].fillna(-1.0)[:]
basic_scaled = preprocessing.scale(basic_features)
basic_scaled_df = pd.DataFrame(basic_scaled, columns=basic_columns)
all_features_df = all_features_df.drop(basic_columns, axis=1).join(basic_scaled_df)

# --------------地址 省---------------
# --------------var22 --------------
text_features = all_features_df[['var' + str(i) for i in range(20, 26, 1)]].fillna('空值')
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
all_features_df = all_features_df.drop(['var' + str(i) for i in range(20, 26, 1)], axis=1).join(tfidf_var_sheng)


# 替换var19的同义词
all_features_df['var19'] = all_features_df['var19'].apply(lambda s: '公司受雇员工' if s=='厨师' or
    s=='公司受雇店长' or s=='负责人' or s=='副店长' or s=='部长' or  s=='公司员工' or   s=='一般员工' or
    s=='普通员工' or s=='受雇员工'  or s=='品质主管' or s=='生产主管' or s=='前台' or s=='司机' or s=='设计师' or
    s=='拓展部经理' or s=='经理' or  s=='快递' or  s=='快递员' or  s=='高管' or  s=='销售管理' or
    s=='销售部长' or s=='销售经理' or s=='员工' or s=='技工' or s=='技术员' or s=='经理' or s=='后厨' or
    s=='厨房' else s)
all_features_df['var19'] = all_features_df['var19'].apply(lambda s: '私企公司职工' if s=='职工' or
    s=='单位职工' or s=='民营事业单位职工' else s)
all_features_df['var19'] = all_features_df['var19'].apply(lambda s: '自由职业者' if s=='失业' or
    s=='失业人员' or s=='家庭主妇' else s)
all_features_df['var19'] = all_features_df['var19'].apply(lambda s: '自主创业人员' if s=='创业人员' or
    s=='自雇创业人员' else s)

# one_hot处理
all_features_df['var19'] = all_features_df['var19'].astype('category')
dummies = pd.get_dummies(all_features_df['var19'], prefix='var19')
all_features_df = all_features_df.drop('var19', axis=1).join(dummies)

basic_columns = ['var2', 'var3', 'var4', 'var5', 'var6', 'var7',
                 'var8', 'var9', 'var10', 'var11', 'var12', 'var13',
                 'var15', 'var16', 'var17', 'var18', 'var19_公务员', 'var19_公司受雇员工',
                 'var19_军人', 'var19_农林牧副渔人员', 'var19_国企事业单位职工', 'var19_私企公司职工',
                 'var19_自主创业人员', 'var19_自由职业者', 'log_var1', 'log_var14', 'log_var6']

def count_call_number(vars):
    vars = vars.fillna('空值')
    curr_var = 'var124'
    vars[curr_var] = vars[curr_var].apply(lambda str: 0 if str == '空值' else len(str.strip().split('@')))
    curr_var = 'var125'
    vars[curr_var] = vars[curr_var].apply(lambda str: str.replace(' ', ''))
    vars[curr_var] = vars[curr_var].apply(lambda str: ' '.join(str.strip().split('@')))
    vars[curr_var] = vars[curr_var].apply(lambda str: ' '.join(str.strip().split('；')))
    vars[curr_var] = vars[curr_var].apply(lambda str: 0 if str == '空值' else len(str.strip().split(' ')))
    curr_var = 'var126'
    vars[curr_var] = vars[curr_var].apply(lambda str: str.replace(' ', ''))
    vars[curr_var] = vars[curr_var].apply(lambda str: ' '.join(str.strip().split('@')))
    vars[curr_var] = vars[curr_var].apply(lambda str: ' '.join(str.strip().split('；')))
    vars[curr_var] = vars[curr_var].apply(lambda str: 0 if str == '空值' else len(str.strip().split(' ')))
    return vars.rename(columns={'var124': 'var124_num', 'var125': 'var125_num', 'var126': 'var126_num'})


# ==============电话信息===============
call_columns = ['var' + str(i) for i in range(26, 134, 1)]
call_columns = pd.Index(call_columns).drop(['var45', 'var124', 'var125', 'var126'])

all_features_df['var26'] = all_features_df['var26'].apply(lambda x: log(x))
all_features_df['var28'] = all_features_df['var28'].apply(lambda x: log(x))
all_features_df['var29'] = all_features_df['var29'].apply(lambda x: log(x))
all_features_df['var30'] = all_features_df['var30'].apply(lambda x: log(x))
all_features_df['var31'] = all_features_df['var31'].apply(lambda x: log(x))
all_features_df['var37'] = all_features_df['var37'].apply(lambda x: log(x))
all_features_df['var38'] = all_features_df['var38'].apply(lambda x: log(x))
all_features_df['var39'] = all_features_df['var39'].apply(lambda x: log(x))
all_features_df['var40'] = all_features_df['var40'].apply(lambda x: log(x))
all_features_df['var48'] = all_features_df['var48'].apply(lambda x: log(x))
all_features_df['var49'] = all_features_df['var49'].apply(lambda x: log(x))
all_features_df['var50'] = all_features_df['var50'].apply(lambda x: log(x))
all_features_df['var51'] = all_features_df['var51'].apply(lambda x: log(x))
all_features_df['var54'] = all_features_df['var54'].apply(lambda x: log(x))
all_features_df['var56'] = all_features_df['var56'].apply(lambda x: log(x))
all_features_df['var57'] = all_features_df['var57'].apply(lambda x: log(x))
all_features_df = all_features_df.drop('var58', axis=1)
all_features_df['var59'] = all_features_df['var59'].apply(lambda x: log(x))
all_features_df['var60'] = all_features_df['var60'].apply(lambda x: log(x))
all_features_df['var61'] = all_features_df['var61'].apply(lambda x: log(x))
all_features_df['var62'] = all_features_df['var62'].apply(lambda x: log(x))
all_features_df['var63'] = all_features_df['var63'].apply(lambda x: log(x))
all_features_df['var65'] = all_features_df['var65'].apply(lambda x: log(x))
all_features_df['var66'] = all_features_df['var66'].apply(lambda x: log(x))
all_features_df['var67'] = all_features_df['var67'].apply(lambda x: log(x))
all_features_df['var68'] = all_features_df['var68'].apply(lambda x: log(x))
all_features_df['var69'] = all_features_df['var69'].apply(lambda x: log(x))
all_features_df['var70'] = all_features_df['var70'].apply(lambda x: log(x) if x != 0 else 0)
all_features_df['var71'] = all_features_df['var71'].apply(lambda x: log(x))
all_features_df['var72'] = all_features_df['var72'].apply(lambda x: log(x) if x != 0 else 0)
all_features_df['var74'] = all_features_df['var74'].apply(lambda x: log(x) if x != 0 else 0)
all_features_df['var76'] = all_features_df['var76'].apply(lambda x: log(x) if x != 0 else 0)
all_features_df['var78'] = all_features_df['var78'].apply(lambda x: log(x) if x != 0 else 0)
all_features_df['var80'] = all_features_df['var80'].apply(lambda x: log(x) if x != 0 else 0)
all_features_df['var82'] = all_features_df['var82'].apply(lambda x: log(x) if x != 0 else 0)
all_features_df['var83'] = all_features_df['var83'].apply(lambda x: log(x) if x != 0 else 0)
all_features_df['var84'] = all_features_df['var84'].apply(lambda x: log(x) if x != 0 else 0)
all_features_df['var86'] = all_features_df['var86'].apply(lambda x: log(x) if x != 0 else 0)
all_features_df['var89'] = all_features_df['var89'].apply(lambda x: log(x) if x != 0 else 0)
all_features_df['var90'] = all_features_df['var90'].apply(lambda x: log(x) if x != 0 else 0)
all_features_df['var92'] = all_features_df['var92'].apply(lambda x: log(x) if x != 0 else 0)
all_features_df['var93'] = all_features_df['var93'].apply(lambda x: log(x) if x != 0 else 0)
all_features_df['var95'] = all_features_df['var95'].apply(lambda x: log(x) if x != 0 else 0)
all_features_df['var96'] = all_features_df['var96'].apply(lambda x: log(x) if x != 0 else 0)
all_features_df['var98'] = all_features_df['var98'].apply(lambda x: log(x) if x != 0 else 0)
all_features_df['var99'] = all_features_df['var99'].apply(lambda x: log(x) if x != 0 else 0)
all_features_df['var101'] = all_features_df['var101'].apply(lambda x: log(x) if x != 0 else 0)
all_features_df['var102'] = all_features_df['var102'].apply(lambda x: log(x) if x != 0 else 0)
all_features_df['var104'] = all_features_df['var104'].apply(lambda x: log(x) if x != 0 else 0)
all_features_df['var107'] = all_features_df['var107'].apply(lambda x: log(x) if x != 0 else 0)
all_features_df['var108'] = all_features_df['var108'].apply(lambda x: log(x) if x != 0 else 0)
all_features_df['var109'] = all_features_df['var109'].apply(lambda x: log(x) if x != 0 else 0)
all_features_df['var116'] = all_features_df['var116'].apply(lambda x: log(x) if x != 0 else 0)
all_features_df['var117'] = all_features_df['var117'].apply(lambda x: log(x) if x != 0 else 0)
all_features_df['var118'] = all_features_df['var118'].apply(lambda x: log(x) if x != 0 else 0)
all_features_df['var119'] = all_features_df['var119'].apply(lambda x: log(x) if x != 0 else 0)
all_features_df['var120'] = all_features_df['var120'].apply(lambda x: log(x) if x != 0 else 0)
all_features_df['var121'] = all_features_df['var121'].apply(lambda x: log(x) if x != 0 else 0)
all_features_df['var127'][all_features_df['var127'] > 1000] = 500
all_features_df['var128'] = all_features_df['var128'].apply(lambda x: log(x) if x != 0 else 0)
all_features_df['var130'][all_features_df['var130'] > 300] = 200
all_features_df['var131'][all_features_df['var131'] > 200] = 200

# 分布比较奇怪的
all_features_df['var32'] = all_features_df['var32'].apply(lambda x: log(x) if x != 0 else 0)
all_features_df['var33'] = all_features_df['var33'].apply(lambda x: log(x) if x != 0 else 0)
all_features_df['var34'] = all_features_df['var34'].apply(lambda x: log(x) if x != 0 else 0)
all_features_df['var53'] = all_features_df['var53'].apply(lambda x: log(x) if x != 0 else 0)
all_features_df['var55'] = all_features_df['var55'].apply(lambda x: log(x) if x != 0 else 0)
all_features_df['var64'] = all_features_df['var64'].apply(lambda x: log(x) if x != 0 else 0)
all_features_df['var73'] = all_features_df['var73'].apply(lambda x: log(x) if x != 0 else 0)
all_features_df['var75'] = all_features_df['var75'].apply(lambda x: log(x) if x != 0 else 0)
all_features_df['var77'] = all_features_df['var77'].apply(lambda x: log(x) if x != 0 else 0)
all_features_df['var79'] = all_features_df['var79'].apply(lambda x: log(x) if x != 0 else 0)
all_features_df['var81'] = all_features_df['var81'].apply(lambda x: log(x) if x != 0 else 0)
all_features_df['var85'] = all_features_df['var85'].apply(lambda x: log(x) if x != 0 else 0)
all_features_df['var87'] = all_features_df['var87'].apply(lambda x: log(x) if x != 0 else 0)
all_features_df['var88'] = all_features_df['var88'].apply(lambda x: log(x) if x != 0 else 0)
all_features_df['var113'] = all_features_df['var113'].apply(lambda x: log(x) if x != 0 else 0)


# 变为数值型，填充空值
call_columns = call_columns.drop('var58')
call_features = all_features_df[call_columns][:]
call_features = call_features.join(count_call_number(all_features_df[['var124', 'var125', 'var126']][:]))
call_features['var47'] = call_features['var47'].fillna(call_features['var47'].median())
call_features = call_features.fillna(call_features.median())

call_scaled = preprocessing.scale(call_features)
call_scaled_df = pd.DataFrame(call_scaled, columns=call_features.columns)

all_features_df = all_features_df.drop(call_columns, axis=1) \
    .drop(['var45', 'var124', 'var125', 'var126'], axis=1).join(call_scaled_df)

# ================信用信息=========================
credit_columns = ['var' + str(i) for i in range(134, 533, 1)]
credit_features = all_features_df[credit_columns]

credit_features = credit_features.drop('var135', axis=1) \
    .drop(['var' + str(i) for i in range(489, 533, 1)], axis=1)
credit_features = credit_features.fillna(-1.)

credit_scaled = preprocessing.scale(credit_features)
credit_scaled_df = pd.DataFrame(credit_scaled, columns=credit_features.columns)
all_features_df = all_features_df.drop(['var' + str(i) for i in range(134, 533, 1)], axis=1).join(credit_scaled_df)

def combine(str,all_features_df):
    if '+' in str:
        vars = str.split('+')
        return pd.DataFrame({str: all_features_df[vars[0]] + all_features_df[vars[1]]})
    elif '-' in str:
        vars = str.split('-')
        return pd.DataFrame({str: all_features_df[vars[0]] - all_features_df[vars[1]]})
    elif '*' in str:
        vars = str.split('*')
        return pd.DataFrame({str: all_features_df[vars[0]] * all_features_df[vars[1]]})
    return None

# 特征组合
print('组合特征...')
# combine_list = []
combine_list = ['var19_私企公司职工+var250', 'var130+var_sheng_10', 'var112+var19_私企公司职工',
       'var19_公司受雇员工*var4', 'var113-var26', 'var112*var55',
       'log_var1-var250', 'var112-log_var1', 'var112-var_sheng_10',
       'log_var6-var94', 'var128-var2', 'var19_私企公司职工-var13',
       'var55*var130', 'var369-var448', 'var4-var428', 'var112-log_var14',
       'log_var14-var327', 'var100+var40', 'var100-var2', 'var67*var109',
       'var209-var327', 'var37*var93', 'var_sheng_10+var_sheng_21',
       'log_var6+var2', 'var327+var7', 'var128-var72', 'var209*var327',
       'var40-var39', 'var19_公司受雇员工+var130', 'var112+var2', 'var57+var82',
       'var130*var13', 'var209-var266', 'log_var14*var19_私企公司职工',
       'var55-var57', 'var94*var10', 'log_var6+var117', 'var110*var47',
       'var19_私企公司职工*var428', 'var19_私企公司职工+var_sheng_11',
       'log_var1+var353', 'log_var6-var156', 'var12*var19_公司受雇员工',
       'var26+var36', 'var57-var61', 'var19_公司受雇员工*var13', 'var250+var156',
       'var353-var103', 'var55-var19_公司受雇员工', 'var66*var4',
       'var19_私企公司职工-var47', 'var94*var97', 'var103+var122', 'var27+var13',
       'var94*var76', 'log_var6+var39', 'var266-var286', 'var61-var56',
       'var86-var127', 'var94-var91', 'var104*var67', 'var128*var99',
       'log_var14*var19_公司受雇员工', 'var57-var99', 'var26-var327',
       'var353-var2', 'log_var6-var13', 'var112-var55', 'var66-var39',
       'log_var1*var4', 'var88-var33', 'var57-var109', 'log_var14*var130',
       'var40*var61', 'var353-var369', 'var19_私企公司职工*var130',
       'log_var14*var110', 'var19_公司受雇员工+var353', 'var94+var12',
       'var112+var103', 'var86-var82', 'var112-var100', 'var4+var7',
       'var353-var96', 'var88*var327', 'var86*var36', 'var4*var7',
       'var26-var57', 'var44+var327', 'var128-var286', 'var26-var36',
       'var26-var91', 'var44*var369', 'var117+var96', 'var94-var55',
       'var19_公司受雇员工+var7', 'var112*var82', 'var127+var_sheng_11',
       'var12*var_sheng_11', 'var4*var428', 'var128-var81', 'var97*var7',
       'var10+var67', 'var37*var448', 'var209-var448', 'log_var6-var327',
       'var12-var369', 'var66+var267', 'log_var1+var156',
       'log_var1+var19_公司受雇员工', 'var128+var353', 'var209*var41',
       'var4+var117', 'var55-var128', 'var55-var113', 'log_var1+var57',
       'var113*var448', 'log_var6*var428', 'var19_私企公司职工*var91',
       'var130*var267', 'log_var1+var61', 'var250+var122', 'var266-var448',
       'var12-var2', 'var57-var96', 'var209*var103', 'var266-var327',
       'var26*var7', 'var72*var88', 'var128+var266', 'var26+var127',
       'var10*var88', 'var100-var103', 'var267+var93', 'var40+var267',
       'var27*var44', 'var94-var103', 'var100-var122', 'log_var6+var36',
       'var156*var153', 'var26+var2', 'var4+var369', 'var100-var97',
       'log_var1*var7', 'var113*var266', 'var113-var61', 'var100+var12',
       'var19_公司受雇员工*var103', 'var97*var44', 'var12*var67', 'var12*var64',
       'var112*var27', 'var113-var39', 'var112-var128', 'var104-var99',
       'var19_私企公司职工-var327', 'var2+var61', 'var55-var267',
       'var112-var19_公司受雇员工', 'var113+var353', 'var94+var209',
       'var327-var369', 'var86+var76', 'var112+var250',
       'var19_公司受雇员工+var_sheng_21', 'var40+var327', 'var112-var66',
       'var104*var327', 'var110*var64', 'var2+var36', 'var100*var110',
       'var428-var369', 'var70-var250', 'var94+var2', 'var112+log_var6',
       'var112-var156', 'var128-var84', 'var19_公司受雇员工+var41',
       'var100+var57', 'var55-var56', 'var100*var4', 'var39*var64',
       'var26-var96', 'var113*var103', 'var86*var82', 'var12+var96',
       'var86*var117', 'var267-var82', 'var100+var56', 'var93*var153']


for str in combine_list:
    scaled_combine = pd.DataFrame(preprocessing.scale(combine(str,all_features_df)),
                                  columns=[str])
    all_features_df = all_features_df.join(scaled_combine)


print("存储数据...")
print('位置：../../data/data_7/')
train_feat = all_features_df[:24309]
train_df = train_feat.join(y_train)

test_feat = all_features_df[24309:]
test_df = test_feat
#
train_df.to_csv('../../data/data_7/train_combined.csv', encoding='gbk', index=False)
test_df.to_csv('../../data/data_7/test_combined.csv', encoding='gbk', index=False)


# print('导入算法包...')
# import time
# import pandas as pd
# import xgboost as xgb
# import sklearn
# from xgboost.sklearn import XGBClassifier
# time_begin = time.time()
#
# predictors = train_df.columns.drop(['id','target'])
# target = 'target'
#
# print('数据量：', train_df.shape)
# # --------------END 读取数据------------------
#
#
# def modelfit(model, dtrain, useTrainCV=True, cv_fold=5, early_stopping_rounds=50):
#     if useTrainCV:
#         print('交叉验证自动设定迭代次数...')
#         xgb_param = model.get_xgb_params()
#         xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
#         # cv:获取最优的boost_round?
#         cvresult = xgb.cv(xgb_param, xgtrain,
#                           num_boost_round=xgb_param['n_estimators'],
#                           nfold=cv_fold,
#                           metrics='auc',
#                           early_stopping_rounds=early_stopping_rounds,
#                           seed=36,
#                           callbacks=[xgb.callback.print_evaluation(show_stdv=True)])
#         print('cv完成 n_estimators：', cvresult.shape[0])
#         model.set_params(n_estimators=cvresult.shape[0])
#     # fit
#     print('模型拟合中...')
#     model.fit(dtrain[predictors],
#             dtrain[target],
#             eval_metric='auc',
#             verbose=True)
#     # predict
#     dtrain_predictions = model.predict(dtrain[predictors])
#     dtrain_predprob = model.predict_proba(dtrain[predictors])[:, 1]
#     # print model report
#     print('\nModel Report')
#     print('Accuracy:%f' % sklearn.metrics.accuracy_score(dtrain[target].values, dtrain_predictions))
#     print('AUC Score (Train): %f' % sklearn.metrics.roc_auc_score(dtrain[target], dtrain_predprob))
#     # plot
#     feat_imp = pd.Series(model.feature_importances_, index=predictors).sort_values(ascending=False)
#     # feat_imp.plot(kind='bar', title='Feature Importances')
#     # plt.ylabel('Feature Importance Score')
#     return feat_imp
#
#
# # ----------当前最优参数----------------
# def get_tuned_xgb():
#     return XGBClassifier(
#         learning_rate=0.1,  # eta
#         n_estimators=1000,
#         # max_depth=3,
#         # min_child_weight=1,
#         # gamma=0.0,
#         # subsample=0.8,
#         # colsample_bytree=0.7,
#         # reg_alpha=1e-08,        # alpha
#         # reg_lambda=1,       # lambda
#         objective='binary:logistic',
#         nthread=-1,
#         seed=36)
#
# xgb_model = get_tuned_xgb()
#
#
# # -----------只用当前参数训练一个模型-------------
# print('训练模型中...')
#
# feat_imp = modelfit(xgb_model, train_df, useTrainCV=True)
#
# # -----------END 只用当前参数训练一个模型---------
#
# print("存储变量重要性：")
# print('位置：./feat_imp/')
# feat_imp.to_csv('./feat_imp/combines_feat_imp_xgb_190.csv', encoding='gbk')
#
#
# time_spend = time.time() - time_begin
# print('\n运行时间：%d 秒，约%d分钟\n' % (time_spend, time_spend // 60))
