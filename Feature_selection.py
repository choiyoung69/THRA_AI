#모듈 입력
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#filter method dataframe 만드는 함수
def filter_method_df(method, n):
    test = SelectKBest(score_func=method, k=n)
    fit = test.fit(X, y)
    f_order = np.argsort(-fit.scores_)
    return raw_df.columns[f_order]

#k-fold 10번 평균
def average_score(model, X, y):
    scores = 0
    for i in range(10):
        kf = KFold(n_splits=5, shuffle=True, random_state=i)
        score = cross_val_score(model, X, y, cv=kf)
        scores += np.mean(score)
    return scores/10

#filter method
def filter_method_score(model, X, filter_df):
    for i in range(1, X.shape[1] + 1):
        fs = filter_df[0:i]
        X_selected = X[fs]
        scores = average_score(model, X_selected, y)
        print(fs.tolist())
        print(np.round(scores, 3))

raw_df = pd.read_csv("C:\\Users\\young\\OneDrive\\바탕 화면\\THRA\\230629_THRA_Raw_data_정리본.csv")

#encoding
number = LabelEncoder()
raw_df['Gender'] = number.fit_transform(raw_df['Gender']).astype('int')

#NULL 데이터 확인 -- 50개 넘으면 제거하고 filter method 이용한 이용한 다음에 결측값 처리하기
raw_df.isnull().sum()
#null값이 반 이상을 차지하는 column 제거
raw_df = raw_df.drop(['Post_Hip ER_MMT', 'Post_Hip IR_MMT'], axis=1)

#오류가 나는 열 처리(value!에러와 str 값 섞인 열)
raw_df.loc[123, 'Days(OP-Cause)']
raw_df = raw_df.replace('#VALUE!', np.nan)

for i in range(len(raw_df)):
    raw_df.loc[i, 'Days(OP-Cause)'] = float(raw_df.loc[i, 'Days(OP-Cause)'])

#null 값은 중앙값으로 대체
raw_df = raw_df.fillna(raw_df.median())

######################################################################
###########################Feature Selection##########################
######################################################################
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

raw_df = raw_df.drop(['Group1', 'Group2', 'Group3', 'Group4',
       'Group5(class)', 'Group6'], axis= 1)
X = raw_df.drop(['Discharge state'], axis= 1)
y = raw_df['Discharge state']

y.value_counts()

###Feature Selection 전####
#LogisticRegression --- 0.578
model = LogisticRegression(solver= 'liblinear',max_iter=10000)
mae  = average_score(model, X, y)
print("mae: %0.3f" %mae)

#RandomForestRegression --- 0.164
model = RandomForestRegressor()
mae= average_score(model, X, y)
print("mae: %0.3f" %mae)

#XGBoostClassfier --- 0.020
model = XGBRegressor()
mae = average_score(model, X, y)
print("mae: %0.3f" %mae)

##### 1.filter method
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression

#1) LogisticRegression
model = LogisticRegression(solver='liblinear', max_iter=10000)

#func = f_regression : 0.653
df_fregression = filter_method_df(f_regression, X.shape[1])
filter_method_score(model, X, df_fregression)
logistic_fregression_df = raw_df.loc[:,['Age', 'Gender', 'Pre_Hip flex_MMT', 'Pre_Hip abd_MMT']]
average_score(model, logistic_fregression_df, y)

#func = mutual_info_regression : 0.654
df_mutual = filter_method_df(mutual_info_regression, X.shape[1])
filter_method_score(model, X, df_mutual)
logistic_mutual_df = raw_df.loc[:, ['Gender', 'Pre_Hip abd_PROM', 'Age']]
average_score(model, logistic_mutual_df, y)

#2) RandomForestRegression
model = RandomForestRegressor()

#func = f_regression : 0.217
df_rf_fregression = filter_method_df(f_regression, X.shape[1])
filter_method_score(model, X, df_rf_fregression)
rf_fregression_df = raw_df.loc[:,['Age', 'Gender', 'Pre_Hip flex_MMT', \
    'Pre_Hip abd_MMT', 'Post_Hip flex_MMT', 'Premorbid status', 'Post_Hip add_MMT', \
    'Post_Hip extensor_MMT', 'Pre_Hip extensor_MMT', 'Post_Hip abd_MMT', \
        'Pre_Hip add_MMT', 'Post_Hip flex_PROM', 'Pre_Hip ER_MMT', 'Insert', 
    'Pre_Hip IR_MMT', 'Cause of THRA', 'Post_Hip extensor_PROM', 'Date(IA-OP)', 
    'Pre_Hip ER_PROM', 'Pre_Hip abd_PROM', 'Days(P-bargait-OP)']]
average_score(model, rf_fregression_df, y)

#func = mutual_info_regression : 0.190
df_rf_mutual = filter_method_df(mutual_info_regression, X.shape[1])
filter_method_score(model, X, df_rf_mutual)
rf_mutual_df = raw_df.loc[:, ['Premorbid status', 'Gender', 
'Post_Hip flex_PROM', 'Pre_Hip add_MMT', 'Date(IA-OP)', 'Pre_Hip ER_PROM', \
'Cause of THRA', 'Age', 'Insert', 'Op Cx.', 'Pre_Hip flex_MMT', 'Pre_Hip IR_MMT']]
average_score(model, rf_mutual_df, y)

#3) XGBoost
model = XGBRegressor()

#func = f_regression 
import warnings
warnings.filterwarnings("ignore")
df_xgb_fregression = filter_method_df(f_regression, X.shape[1])
filter_method_score(model, X, df_xgb_fregression)
warnings.simplefilter("default")
xgb_fregression_df = raw_df.loc[:,['Age', 'Gender', 'Pre_Hip flex_MMT', 'Pre_Hip abd_MMT', \
'Post_Hip flex_MMT', 'Premorbid status', 'Post_Hip add_MMT', 'Post_Hip extensor_MMT', \
'Pre_Hip extensor_MMT', 'Post_Hip abd_MMT', 'Pre_Hip add_MMT', 'Post_Hip flex_PROM',\
 'Pre_Hip ER_MMT', 'Insert', 'Pre_Hip IR_MMT', 'Cause of THRA', 'Post_Hip extensor_PROM', \
    'Date(IA-OP)', 'Pre_Hip ER_PROM', 'Pre_Hip abd_PROM', 'Days(P-bargait-OP)']]
average_score(model, xgb_fregression_df, y)

#func = mutual_info_regression 
import warnings
warnings.filterwarnings("ignore")
df_xgb_mutual = filter_method_df(mutual_info_regression, X.shape[1])
filter_method_score(model, X, df_xgb_mutual)
warnings.simplefilter("default")
xgb_mutual_df = raw_df.loc[:,['Age', 'Premorbid status', 'Date(IA-OP)', 'Pre_Hip abd_PROM', \
'Pre_Hip add_MMT', 'Op Cx.', 'Pre_Hip extensor_MMT', 'Pre_Hip ER_MMT', 'Post_Hip extensor_PROM', \
'Pre_Hip flex_MMT', 'Post_Hip flex_PROM', 'Head', 'Gender', 'Pre_Hip IR_PROM', 'Revision', \
'Post_Hip flex_MMT', 'Post_Hip abd_PROM', 'Days(OP-Cause)', 'Pre_Hip flex_PROM', 'Pre_Hip abd_MMT', \
'Osteoporosis', 'Post_Hip abd_MMT', 'Days(OP-GD)', 'Days(P-bargait-OP)', 'Pre_Hip extensor_PROM',\
 'Post_Hip extensor_MMT', 'Cause of THRA', 'Rt./Lt.', 'Post_Hip add_MMT', 'Post_Hip add_PROM', 'Steroid', 'Days(OP-PTx.)']]
average_score(model, xgb_mutual_df, y)


#2. RFE
from sklearn.feature_selection import RFECV

#1) LogisticRegression 
model = LogisticRegression(solver='liblinear', max_iter=10000, random_state=144)

rfecv = RFECV(estimator=model, step=1, cv=5)
fit = rfecv.fit(X, y)
print("num Features: %d" %fit.n_features_)
fs = X.columns[fit.support_].tolist()
print("selected Features: %s" %fs)

acc = average_score(model, X[fs], y)
print("ACC: " + str(acc))

#2) RandomForestRegression 
model = RandomForestRegressor(random_state=100)

rfecv = RFECV(estimator=model, step=1, cv=5)
fit = rfecv.fit(X, y)
print("num Features: %d" %fit.n_features_)
fs = X.columns[fit.support_].tolist()
print("selected Features: %s" %fs)

acc = average_score(model, X[fs], y)
print("ACC: " + str(acc))

#3) XGBoost 
import warnings
warnings.filterwarnings("ignore")

model = XGBRegressor(random_state=1)

rfecv = RFECV(estimator=model, step=1, cv=5)
fit = rfecv.fit(X, y)
print("num Features: %d" %fit.n_features_)
fs = X.columns[fit.support_].tolist()
print("selected Features: %s" %fs)

acc = average_score(model, X[fs], y)
print("ACC: " + str(acc))
warnings.simplefilter("default")

#3. SFS
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import warnings

#1) LogisticRegression --- 0.670
warnings.filterwarnings("ignore")
model = LogisticRegression(solver='liblinear', max_iter=10000, random_state=10)
sfs = SFS(model,
        k_features=X.shape[1],
        verbose=2,
        scoring='accuracy',
        cv = 5)
sfs = sfs.fit(X, y)
sfs.subsets_
X_logi_sfs = X[['Gender', 'Age', 'Pre_Hip flex_MMT', 'Pre_Hip abd_MMT', \
                'Pre_Hip ER_PROM', 'Post_Hip add_PROM', 'Days(P-bargait-OP)']]
acc = average_score(model, X_logi_sfs, y)
print("acc:", acc)
warnings.simplefilter("default")

#2) RandomForestRegression
warnings.filterwarnings("ignore")
model = RandomForestRegressor()
sfs = SFS(model,
        k_features=X.shape[1],
        verbose=2,
        cv = 5)
sfs = sfs.fit(X, y)
sfs.subsets_
X_rf_sfs = X[['Gender', 'Age', 'Premorbid status', 'Cause of THRA', 'Rt./Lt.', 'Days(OP-GD)',\
 'Osteoporosis', 'Steroid', 'Pre_Hip flex_MMT', 'Pre_Hip extensor_MMT', 'Pre_Hip abd_MMT', \
'Pre_Hip add_MMT', 'Pre_Hip IR_MMT', 'Pre_Hip flex_PROM', 'Pre_Hip extensor_PROM', 'Pre_Hip abd_PROM',\
     'Revision', 'Head', 'Op Cx.', 'Duration of PTx.', 'Post_Hip extensor_MMT', 'Post_Hip flex_PROM', 'Days(P-bargait-OP)']]
acc = average_score(model, X_rf_sfs, y)
print("acc:", acc)
warnings.simplefilter("default")

#3) XGBoost
warnings.filterwarnings("ignore")
model = RandomForestRegressor()
sfs = SFS(model,
        k_features=X.shape[1],
        verbose=2,
        scoring='neg_mean_absolute_error',
        cv = 5)
sfs = sfs.fit(X, y)
sfs.subsets_
X_xgb_sfs = X[['Gender', 'Age', 'Pre_Hip flex_MMT', 'Pre_Hip abd_MMT', \
                'Pre_Hip ER_PROM', 'Post_Hip add_PROM', 'Days(P-bargait-OP)']]
acc = average_score(model, X_xgb_sfs, y)
print("acc:", acc)
warnings.simplefilter("default")
