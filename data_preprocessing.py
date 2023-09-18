#모듈 입력
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#IQR 지표를 이용한 이상치 확인
def outlier(data, column):
    q25, q75 = np.quantile(data[column], 0.25), np.quantile(data[column], 0.75)
    iqr = q75 - q25

    cut_off = iqr * 1.5
    lower, upper = q25 - cut_off, q75 + cut_off
    data1 = data[data[column] > upper]     
    data2 = data[data[column] < lower]

    return print(column, '의 이상치 개수는', data1.shape[0] + data2.shape[0], "이다")

raw_df = pd.read_csv("C:\\Users\\young\\OneDrive\\바탕 화면\\THRA\\230629_THRA_Raw_data_정리본.csv")

#각 환자의 discharge state 값 확인 => 이미 discharge state가 존재
raw_df.insert(48, 'state', -1)

raw_df.loc[raw_df['Group1'] == 0, 'state'] = 7
raw_df.loc[(raw_df['Group2'] == 0) & (raw_df['state'] == -1), 'state'] = 6
raw_df.loc[(raw_df['Group3'] == 0) & (raw_df['state'] == -1), 'state'] = 5
raw_df.loc[(raw_df['Group4'] == 0) & (raw_df['state'] == -1), 'state'] = 4 
raw_df.loc[(raw_df['Group5(class)'] == 0) & (raw_df['state'] == -1), 'state'] = 3
raw_df.loc[(raw_df['Group6'] == 0) & (raw_df['state'] == -1), 'state'] = 2
raw_df.loc[raw_df['Group6'] == 1, 'state'] = 1

#데이터 확인 - 성별과 Days(OP-Cause) column을 제외하면 int나 float
raw_df.dtypes

#One-hot encoding
data = {'Gender' : [value for value in raw_df['Gender']]}
df = pd.DataFrame(data)
encoded_df = pd.get_dummies(df, columns=['Gender'])
encoded_df
raw_df.insert(1, 'Gender_F', encoded_df['Gender_F'])
raw_df.insert(2, 'Gender_M', encoded_df['Gender_M'])
raw_df['Gender_F'] = raw_df['Gender_F'].replace({True: 1, False: 0})
raw_df['Gender_M'] = raw_df['Gender_M'].replace({True: 1, False: 0})

raw_df = raw_df.drop(['Gender'], axis=1)

#NULL 데이터 확인 -- 50개 넘으면 제거하고 filter method 이용한 이용한 다음에 결측값 처리하기
raw_df.isnull().sum()
#null값이 반 이상을 차지하는 column 제거
raw_df = raw_df.drop(['Post_Hip ER_MMT', 'Post_Hip IR_MMT'], axis=1)
#나머지 column의 null 값 처리 - 평균으로 채우기
#raw_df.dtypes
#raw_df = raw_df.replace('#VALUE!', pd.NA)

raw_df = raw_df.drop('Days(OP-Cause)', axis=1)
raw_df = raw_df.fillna(raw_df.mean())

#이상치 확인(이상치를 가공해야 할 것 같음 log)
for i in range(len(raw_df.columns)):
    if pd.api.types.is_numeric_dtype(raw_df.iloc[:, i].dtype):
        outlier(raw_df, raw_df.columns[i])

#Permorbid status : 이상치 54개, Osteporosis : 이상치 32개, Days(OP-PTx.) : 이상치 21개

#Logistic Regression, RandomForest, XGBoost
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import explained_variance_score

X = raw_df.drop(['Discharge state', 'Group1', 'Group2', 'Group3', 'Group4',
       'Group5(class)', 'Group6', 'state'], axis= 1)
y = raw_df['Discharge state']

train_X, test_X, train_y, test_y = \
                    train_test_split(X, y, test_size=0.3, random_state=10)

#LogisticRegression
model = LogisticRegression()
model.fit(train_X, train_y)
pred_y = model.predict(test_X)

#LogisticRegression의 정확도: 0.603
acc = accuracy_score(test_y, pred_y)
print('Accurancy: {0:.3f}' .format(acc))

#validation을 어떻게 나누느냐에 따라서 너무 많이 바뀜
#
#RandomForestRegressor 
model = RandomForestRegressor(random_state=10)
model.fit(train_X, train_y)
model.score(test_X, test_y) #0.13

#XGBoost
model = XGBRegressor()
xr = model.fit(train_X, train_y)
model.score(test_X, test_y) #0.03