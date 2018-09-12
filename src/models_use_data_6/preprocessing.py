# encoding: utf-8

import pandas as pd
from sklearn import preprocessing
import sklearn
from math import log
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

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
all_features_df = all_features_df.drop(basic_columns, axis=1) \
    .drop(['var' + str(i) for i in range(20, 26, 1)], axis=1) \
    .join(basic_scaled_df)

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

clf = LinearDiscriminantAnalysis()
clf.fit(basic_features[:y_train.shape[0]], y_train)
basic_ldaed = clf.transform(basic_features)

basic_scaled = preprocessing.scale(basic_ldaed)
basic_scaled_df = pd.DataFrame(basic_scaled, columns=['basic_ldaed'])
all_features_df = all_features_df.join(basic_scaled_df)


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

clf = LinearDiscriminantAnalysis()
clf.fit(call_features[:y_train.shape[0]], y_train)
call_ldaed = clf.transform(call_features)

call_scaled = preprocessing.scale(call_ldaed)
call_scaled_df = pd.DataFrame(call_scaled, columns=['call_ldaed'])
all_features_df = all_features_df.join(call_scaled_df)

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

clf = LinearDiscriminantAnalysis()
clf.fit(credit_features[:y_train.shape[0]], y_train)
credit_ldaed = clf.transform(credit_features)

credit_scaled = preprocessing.scale(credit_ldaed)
credit_scaled_df = pd.DataFrame(credit_scaled, columns=['credit_ldaed'])
all_features_df = all_features_df.join(credit_scaled_df)

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
combine_list =['credit_ldaed+basic_ldaed', 'call_ldaed+credit_ldaed',
       'credit_ldaed+var19_公司受雇员工', 'var19_私企公司职工*var130',
       'var19_私企公司职工+var112', 'var19_公司受雇员工*var4', 'var19_私企公司职工+var250',
       'call_ldaed-var19_私企公司职工', 'call_ldaed+var19_公司受雇员工',
       'var112*var55', 'credit_ldaed-var19_私企公司职工', 'var19_私企公司职工+var131',
       'credit_ldaed+var266', 'var209-var266', 'log_var1-var250',
       'var130+var47', 'call_ldaed+basic_ldaed',
       'basic_ldaed+var19_公司受雇员工', 'var130*var13', 'var19_公司受雇员工*var12',
       'var130*var55', 'var112-log_var14', 'var112-var6',
       'basic_ldaed-var19_私企公司职工', 'var4*var114', 'var112-log_var1',
       'var19_私企公司职工*var428', 'basic_ldaed-var4', 'var66*var13',
       'var103-var80', 'log_var6*var56', 'credit_ldaed-log_var6',
       'var4-var428', 'var4+var250', 'var209*var286', 'var91+var6',
       'credit_ldaed+var428', 'call_ldaed-var4', 'basic_ldaed-var90',
       'var113*var103', 'call_ldaed+var113', 'basic_ldaed-var112',
       'var66*var103', 'basic_ldaed+var40', 'var66*var131', 'var88*var119',
       'basic_ldaed+var37', 'log_var14-var267', 'var112*log_var14',
       'var113*var6', 'var19_私企公司职工*log_var14', 'call_ldaed*var19_公司受雇员工',
       'var37*var93', 'call_ldaed-log_var6', 'var428-var266',
       'var114*var56', 'call_ldaed+var93', 'var19_私企公司职工-var13',
       'var113-var93', 'var19_私企公司职工*var91', 'log_var14+var267',
       'credit_ldaed+var93', 'var130*var99', 'log_var6*var65',
       'var70*var128', 'var70*var80', 'call_ldaed-var103', 'var112-var90',
       'var6*var117', 'var122*var56', 'call_ldaed-var44',
       'call_ldaed*var125_num', 'log_var14*var103', 'var209*var267',
       'var12*var122', 'var99-var93', 'var70+var267', 'log_var6-var267',
       'credit_ldaed-var37', 'log_var6*var127', 'var114+var428',
       'log_var14*var88', 'log_var6+var27', 'var267*var100',
       'credit_ldaed+var13', 'log_var6*var66', 'var70-var13',
       'var19_公司受雇员工*log_var14', 'var12*var286', 'var127+var99',
       'log_var6+var209', 'var99*var65', 'log_var6-var10',
       'credit_ldaed*var19_公司受雇员工', 'var112+var250', 'var12*var10',
       'credit_ldaed+var12', 'var113*var127', 'var70-var40',
       'var127*var27', 'var19_公司受雇员工*var286', 'var19_公司受雇员工-var37',
       'var99*var128', 'var70+var428', 'log_var14-var6',
       'var19_公司受雇员工*var56', 'credit_ldaed+var4', 'var19_公司受雇员工+var65',
       'var12+var266', 'basic_ldaed*log_var14', 'var55-var93',
       'var91-var114', 'var4*var128', 'var4-var100', 'var91-var27',
       'basic_ldaed-var122', 'log_var1-var113', 'var86-var286',
       'call_ldaed*var55', 'var55*var80', 'var19_私企公司职工+log_var1',
       'var86+var93', 'var286-var266', 'var4*log_var14', 'var10*var66',
       'var100+var44', 'var286*var128', 'call_ldaed+var86',
       'var286*var266', 'basic_ldaed-var91', 'log_var1+var6',
       'var131*var40', 'call_ldaed-var12', 'var55+var127', 'var86*var100',
       'var86-var88', 'log_var1*log_var1', 'var4-var6', 'var86-var66',
       'basic_ldaed+var153', 'var209*var103', 'var86-var125_num',
       'var4+var125_num', 'var86+var428', 'call_ldaed+var267',
       'var119*var65', 'var88-var131', 'basic_ldaed+var12',
       'var100-var266', 'var130*var93', 'var40+var267', 'log_var14+var10',
       'var153*var209', 'var153-var209', 'var112*var286',
       'log_var14-var10', 'var267-var6', 'var112+var90', 'var40*var100',
       'log_var14*var153']


for str in combine_list:
    scaled_combine = pd.DataFrame(preprocessing.scale(combine(str,all_features_df)),
                                  columns=[str])
    all_features_df = all_features_df.join(scaled_combine)


print("存储数据...")
print('位置：../../data/data_6/')
train_feat = all_features_df[:24309]
train_df = train_feat.join(y_train)

test_feat = all_features_df[24309:]
test_df = test_feat

train_df.to_csv('../../data/data_6/train_combined.csv', encoding='gbk', index=False)
test_df.to_csv('../../data/data_6/test_combined.csv', encoding='gbk', index=False)

