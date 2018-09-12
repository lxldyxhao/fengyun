# -*- coding=utf-8 -*-

print('导入算法包...')
from sklearn.feature_selection import RFECV
import time
import pandas as pd
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
import getpass
from sklearn.model_selection import cross_val_score

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
    # train_df = train_df[:200]
    # step = 100
    pass
print('数据量：', train_df.shape)


# ----------------　特征选择　------------------

class new_RFECV:
    def __init__(self, estimator, step=10, min=100, cv=5, scoring='roc_auc', n_jobs=-1):
        self.step = step
        self.min = min
        self.cv = cv
        self.n_jobs = n_jobs
        self.scoring = scoring
        self.best_score_ = 0
        self.importance = 0
        self.estimator = estimator
        self.current_features = 0
        self.selected_features = 0

    def fit(self, x_train, y_train):
        self.current_features = x_train.columns
        self.selected_features = x_train.columns
        while (self.current_features.shape[0] > self.min):
            score = cross_val_score(self.estimator,
                                    x_train[self.current_features],
                                    y_train,
                                    scoring=self.scoring,
                                    cv=self.cv,
                                    n_jobs=self.n_jobs).mean()
            self.estimator.fit(x_train[self.current_features], y_train)
            print("当前训练特征数：", self.current_features.shape[0], " CV： ", score)
            if self.best_score_ < score:
                self.best_score_ = score
                self.selected_features = self.current_features[:]
                self.importance = pd.Series(self.estimator.booster().get_fscore()) \
                    .sort_values(ascending=False)
                print("更新最佳特征数")

            curr_importance = pd.Series(self.estimator.booster().get_fscore()) \
                .sort_values(ascending=False)

            ##存在没有重要性的变量,优先drop他们
            no_importance_feat = []
            for i in self.current_features:
                if i not in curr_importance.index:
                    no_importance_feat.append(i)

            if no_importance_feat.__len__() == 0:
                print('curr_importance 变量数：', curr_importance.shape[0])
                tobe_drop = curr_importance.tail(self.step).index.values
                print('drop:', len(tobe_drop), '个', tobe_drop)
                self.current_features = self.current_features.drop(tobe_drop)
            else:
                print('curr_importance', curr_importance.shape[0])
                tobe_drop = no_importance_feat[:self.step]
                print('drop no importance:', len(tobe_drop), '个', tobe_drop)
                self.current_features = self.current_features.drop(tobe_drop)

            print('==============================================')
        print('选择特征数：', self.selected_features.shape[0], ' CV :', self.best_score_)


xgb_estimator = XGBClassifier(
    learning_rate=0.05,
    n_estimators=266,
    max_depth=5,
    min_child_weight=5,
    gamma=0.7,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0,
    reg_lambda=1,
    objective='binary:logistic',
    nthread=-1,
    seed=36)

print('XGB交叉验证迭代筛选特征...')
selector = new_RFECV(xgb_estimator, step=step, cv=5, scoring='roc_auc', n_jobs=-1)
selector.fit(train_df[predictors], train_df[target])
#
# 筛选特征
print('筛选最优特征...')
selected_features = pd.Series(selector.selected_features)

# -------------保存筛选特征后的数据--------------

print("存储选择结果中：")
print('位置：../../data/data_3/preprocessed_data_selection_xgb.csv')
selected_features.to_csv('../../data/data_3/preprocessed_data_selection_xgb.csv', encoding='gbk')

# ----------------输出运行时间  --------------------
time_spend = time.time() - time_begin
print('\n运行时间：%d 秒，约%d分钟\n' % (time_spend, time_spend // 60))
