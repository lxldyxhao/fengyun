# -*- coding=utf-8 -*-

import time
import getpass
import pandas as pd
import xgboost as xgb
import sklearn
from sklearn.model_selection import GridSearchCV
from xgboost.sklearn import XGBClassifier
from sklearn import preprocessing
time_begin = time.time()


# ---------------读取数据---------------------
DATA_DIR = '../../data/data_7/'
print('读取数据...\n位置：', DATA_DIR)

train_df = pd.read_csv( DATA_DIR + 'train_standard.csv', encoding='gbk')
test_df = pd.read_csv(DATA_DIR + 'test_standard.csv', encoding='gbk')
y_train = train_df.iloc[:, -1:]
all_features_df = train_df.iloc[:, :-1].append(test_df, ignore_index=True)




def combination(name1, name2, all_features_df):
    combines = pd.DataFrame({
        name1 + '+' + name2: preprocessing.scale(all_features_df[name1] + all_features_df[name2]),
        name1 + '-' + name2: preprocessing.scale(all_features_df[name1] - all_features_df[name2]),
        name1 + '*' + name2: preprocessing.scale(all_features_df[name1] * all_features_df[name2]),
    })

    return combines

combines_df = pd.DataFrame(all_features_df['id'])

var_list = ['var112', 'log_var1', 'log_var14', 'var86', 'var100',
       'var19_私企公司职工', 'log_var6', 'var94', 'var55', 'var128', 'var113',
       'var12', 'var26', 'var19_公司受雇员工', 'var353', 'var70', 'var66',
       'var37', 'var4', 'var2', 'var10', 'var130', 'var127', 'var250',
       'var110', 'var40', 'var104', 'var209', 'var78', 'var27', 'var97',
       'var57', 'var91', 'var13', 'var72', 'var36', 'var267', 'var156',
       'var39', 'var93', 'var61', 'var266', 'var47', 'var103', 'var44',
       'var122', 'var64', 'var_sheng_10', 'var82', 'var67', 'var153',
       'var286', 'var41', 'var81', 'var117', 'var96', 'var56',
       'var_sheng_11', 'var_sheng_21', 'var84', 'var88', 'var33', 'var76',
       'var428', 'var327', 'var369', 'var7', 'var109', 'var99', 'var448']


for i in range(0,len(var_list)):
    for j in range(i,len(var_list)):
        combines_df = combines_df.join(combination(var_list[i], var_list[j], all_features_df))

all_features_df = combines_df

train_feat = all_features_df[:24309]
train_df = train_feat.join(y_train)
test_feat = all_features_df[24309:]
test_df = test_feat
predictors = train_df.columns.drop(['id','target'])
target = 'target'

print('数据量：', train_df.shape)
# --------------END 读取数据------------------


def modelfit(model, dtrain, useTrainCV=True, cv_fold=5, early_stopping_rounds=50):
    if useTrainCV:
        print('交叉验证自动设定迭代次数...')
        xgb_param = model.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        # train_df.shape
        # cv:获取最优的boost_round?
        cvresult = xgb.cv(xgb_param, xgtrain,
                          num_boost_round=xgb_param['n_estimators'],
                          nfold=cv_fold,
                          metrics='auc',
                          early_stopping_rounds=early_stopping_rounds,
                          seed=36,
                          callbacks=[xgb.callback.print_evaluation(show_stdv=True)])
        print('cv完成 n_estimators：', cvresult.shape[0])
        model.set_params(n_estimators=cvresult.shape[0])
    # fit
    print('模型拟合中...')
    model.fit(dtrain[predictors],
            dtrain[target],
            eval_metric='auc',
            verbose=True)
    # predict
    dtrain_predictions = model.predict(dtrain[predictors])
    dtrain_predprob = model.predict_proba(dtrain[predictors])[:, 1]
    # print model report
    print('\nModel Report')
    print('Accuracy:%f' % sklearn.metrics.accuracy_score(dtrain[target].values, dtrain_predictions))
    print('AUC Score (Train): %f' % sklearn.metrics.roc_auc_score(dtrain[target], dtrain_predprob))
    # plot
    feat_imp = pd.Series(model.feature_importances_, index=predictors).sort_values(ascending=False)
    # feat_imp.plot(kind='bar', title='Feature Importances')
    # plt.ylabel('Feature Importance Score')
    return feat_imp


# ----------当前最优参数----------------
def get_tuned_xgb():
    return XGBClassifier(
        learning_rate=0.1,  # eta
        n_estimators=1000,
        # max_depth=3,
        # min_child_weight=8,
        # gamma=0.1,
        # subsample=0.8,
        # colsample_bytree=0.7,
        # reg_alpha=0.9,        # alpha
        # reg_lambda=1,       # lambda
        objective='binary:logistic',
        nthread=-1,
        seed=36)

xgb_model = get_tuned_xgb()

#
# # -----------只用当前参数训练一个模型-------------
print('训练模型中...')

feat_imp = modelfit(xgb_model, train_df, useTrainCV=True)

# # -----------END 只用当前参数训练一个模型---------
#

print("存储变量重要性：")
print('位置：./')
feat_imp.to_csv('./combines_feat_imp_xgb_top50.csv', encoding='gbk')


time_spend = time.time() - time_begin
print('\n运行时间：%d 秒，约%d分钟\n' % (time_spend, time_spend // 60))
