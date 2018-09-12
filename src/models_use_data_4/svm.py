# -*- coding=utf-8 -*-

print('导入算法包...')
import time
import getpass
import pandas as pd
import sklearn
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
time_begin = time.time()


# ---------------读取数据---------------------
DATA_DIR = '../../data/data_4/'
print('读取数据...\n位置：', DATA_DIR)

train_df = pd.read_csv(DATA_DIR + 'train_preprocessed1.csv', encoding='gbk')

# 使用特征选择的结果

predictors = train_df.columns.drop(['id','target'])
target = 'target'

# if getpass.getuser()=='stone':
#     train_df=train_df[:2000]

print('数据量：', train_df.shape)
# --------------END 读取数据------------------

# ----------当前最优参数----------------
def get_tuned_svc():
    return SVC(
        C=10,
        gamma='auto',
        random_state=36,
    )

model = get_tuned_svc()

# ---------------格点搜索----------------------
# 格点搜索参数
param_test = {

}

gsearch = GridSearchCV(
    estimator=model,
    param_grid=param_test,
    scoring='roc_auc',
    iid=False,
    cv=5,
    n_jobs=3,
)

print('正在搜索...')
gsearch.fit(train_df[predictors], train_df[target])

# 输出搜索结果
print('\n当前格点搜索的参数：\n', param_test)
print('\ngsearch.best_params_:', gsearch.best_params_,)
print('\ngsearch.best_score_:', gsearch.best_score_, )

# ---------------END 格点搜索----------------

time_spend = time.time() - time_begin
print('\n运行时间：%d 秒，约%d分钟\n' % (time_spend, time_spend // 60))
