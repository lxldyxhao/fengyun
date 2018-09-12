# encoding: utf-8

import pandas as pd
from sklearn import preprocessing
import sys
import time
import getpass
import pandas as pd
import xgboost as xgb
import sklearn
from xgboost.sklearn import XGBClassifier

time_begin = time.time()

# 读取数据
print("读取数据中：")
print('位置：../../data/data_5/train_preprocessed2.csv & test_preprocessed2.csv')

DATA_DIR = '../../data/data_5/'
original_train_df = pd.read_csv(DATA_DIR + 'train_preprocessed2.csv', encoding='gbk')
original_test_df = pd.read_csv(DATA_DIR + 'test_preprocessed2.csv', encoding='gbk')

y_train = original_train_df.iloc[:, -1:]
all_features_df = original_train_df.iloc[:, :-1].append(original_test_df, ignore_index=True)

# 特征选择的结果
original_feat_imp = pd.read_csv(DATA_DIR + 'feat_imp_xgb.csv', encoding='gbk',
                       header=None, names=['var', 'is_useful'])


def modelfit(model, dtrain, useTrainCV=True, cv_fold=5, early_stopping_rounds=50):
    if useTrainCV:
        print('交叉验证自动设定迭代次数...')
        xgb_param = model.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
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
              eval_metric='auc')
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


def combination(name1, name2, all_features_df):
    combines = pd.DataFrame({
        name1 + '+' + name2: all_features_df[name1] + all_features_df[name2],
        name1 + '-' + name2: all_features_df[name1] - all_features_df[name2],
        name1 + '*' + name2: all_features_df[name1] - all_features_df[name2],
    })
    return all_features_df.join(combines)


first_var = 1

while (True):

    first_var = first_var + 10
    var_list = original_feat_imp[first_var: first_var + 10]['var'].values

    y_train = original_train_df.iloc[:, -1:]
    all_features_df = original_train_df.iloc[:, :-1].append(original_test_df, ignore_index=True)

    try:
        # 重要性前十名排列组合,还有通话数量
        print('排列组合...')
        print('first_var 当前排名第', first_var, '的变量')

        file = open('log.txt', 'a')
        old = sys.stdout
        sys.stdout = file
        print('排列组合...')
        print('first_var 当前排名第', first_var, '的变量')


        for name1 in var_list:
            for name2 in var_list:
                all_features_df = combination(name1, name2, all_features_df)

        # 丢掉通话记录
        all_features_df = all_features_df.drop(['var12456_' + str(i) for i in range(0, 200)], axis=1)

        # 数值型数据标准化,使用标准的scaler，以后可以有不同的处理方法
        print('标准化...')

        all_features_df = all_features_df.fillna(0)[:]
        X_scaled = preprocessing.scale(all_features_df)
        all_features_df = pd.DataFrame(X_scaled, columns=all_features_df.columns)

        # ---------保存预处理后的数据----------------
        train_feat = all_features_df[:24309]
        train_df = train_feat.join(y_train)

        test_feat = all_features_df[24309:]
        test_df = test_feat

        # ============================================

        predictors = train_df.columns.drop(['id', 'target'])
        target = 'target'

        print('数据量：', train_df.shape)


        # ----------当前最优参数----------------
        def get_tuned_xgb():
            return XGBClassifier(
                learning_rate=0.1,  # eta
                n_estimators=1000,
                objective='binary:logistic',
                nthread=-1,
                seed=36)


        xgb_model = get_tuned_xgb()

        # -----------只用当前参数训练一个模型-------------
        print('训练模型中...')
        feat_imp = modelfit(xgb_model, train_df, useTrainCV=True)

        # -----------END 只用当前参数训练一个模型---------

        sys.stdout=old
        file.close()

        print("存储变量重要性：")
        print('位置：../../data/data_5/')
        feat_imp.to_csv('../../data/data_5/feat_imp_xgb' + str(first_var) + '.csv', encoding='gbk')

        time_spend = time.time() - time_begin
        print('\n运行时间：%d 秒，约%d分钟\n' % (time_spend, time_spend // 60))

    except Exception as e:
        print(e)
        continue
