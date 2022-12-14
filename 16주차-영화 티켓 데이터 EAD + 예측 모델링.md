# 16주차-영화 티켓 데이터 EAD + 예측 모델링

활용 데이터(캐글 제공)

[cinemaTicket_Ref.csv](https://github.com/bjh0507/Study/blob/main/%EC%95%84%EB%A7%88%EC%A1%B4%EB%8D%B0%EC%9D%B4%ED%84%B0_EDA/cinemaTicket_Ref.csv)

## **데이터 확인**

```python
import pandas as pd

df = pd.read_csv('/content/cinemaTicket_Ref.csv')
df.info()
```

![Untitled](https://github.com/bjh0507/Study/blob/main/%EC%95%84%EB%A7%88%EC%A1%B4%EB%8D%B0%EC%9D%B4%ED%84%B0_EDA/10-1.png)

**결측치 확인 & 처리**

```python
df.isnull().sum()
```

![Untitled](https://github.com/bjh0507/Study/blob/main/%EC%95%84%EB%A7%88%EC%A1%B4%EB%8D%B0%EC%9D%B4%ED%84%B0_EDA/10-2.png)

- 총 2개 컬럼에서 125개씩 결측치가 존재한다는 것을 파악

```python
mean1 = df['occu_perc'].mean()
mean2 = df['capacity'].mean()
df['occu_perc'].fillna(mean1, inplace=True)
df['capacity'].fillna(mean2, inplace=True)
df.isnull().sum()
```

![Untitled](https://github.com/bjh0507/Study/blob/main/%EC%95%84%EB%A7%88%EC%A1%B4%EB%8D%B0%EC%9D%B4%ED%84%B0_EDA/10-3.png)

- 결측 수가 많지 않기 때문에 각 컬럼별 평균 값으로 결측치 대체

---

## 데이터 시각화

**월별 상영 횟수 시각화**

```python
df_count = df.groupby('month')['show_time'].sum().reset_index()
df_count['month'] = df_count['month'].astype('str').str.split('.').str[0]

import plotly.express as px

fig = px.bar(df_count, x='month', y='show_time', color='month', text='show_time', title='월별 상영 횟수 시각화')
fig.show()
```

![Untitled](https://github.com/bjh0507/Study/blob/main/%EC%95%84%EB%A7%88%EC%A1%B4%EB%8D%B0%EC%9D%B4%ED%84%B0_EDA/10-4.png)

- 4,5,7,8,10월달에 영화 상영 수가 많음을 알 수 있음

**월별 판매 티켓 비율 시각화**

```python
df3 = df.groupby('month')['tickets_sold'].sum().reset_index()
df3['month'] = df3['month'].astype('str').str.split('.').str[0] + '월'
fig = px.pie(df3, values='tickets_sold', names = 'month', color = 'month', title='월별 판매 티켓 수 시각화')
fig.show()
```

![Untitled](https://github.com/bjh0507/Study/blob/main/%EC%95%84%EB%A7%88%EC%A1%B4%EB%8D%B0%EC%9D%B4%ED%84%B0_EDA/10-5.png)

- 상영횟수가 많은 달에 티켓 판매 비율이 높음을 알 수 있음
- 4,5,6,7,10월 중 4월달이 영화 상영횟수가 가장 낮지만 판매 티켓 비율을 가장 높음을 알 수 있음 → 4월달에 영화 소비하는 관람객이 더 많음을 의미

**월별 수익 시각화**

```python
df4 = df.groupby('month')['total_sales'].sum().reset_index()
df4['month'] = df4['month'].astype('str').str.split('.').str[0]+'월'

fig = px.line(df4, x='month', y='total_sales', title='월별 수익 시각화')
fig.show()
```

![Untitled](https://github.com/bjh0507/Study/blob/main/%EC%95%84%EB%A7%88%EC%A1%B4%EB%8D%B0%EC%9D%B4%ED%84%B0_EDA/10-6.png)

- 4월달에 수익이 가장 높으며, 영화 상영수가 가장 많은 5월달이 생각보다 낮은 수익을 보임

**영화 코드별 상영 횟수**

*어떤 영화가 가장 많이 상영 되었는지 파악해보고자 함*

```python
df['film_code'] = df['film_code'].astype('str')
df5 = df.groupby('film_code')['show_time'].sum().reset_index()
fig = px.bar(df5, x='film_code', y='show_time', color='film_code', text='show_time', title='영화 코드별 상영 횟수 시각화')
fig.show()
```

![Untitled](https://github.com/bjh0507/Study/blob/main/%EC%95%84%EB%A7%88%EC%A1%B4%EB%8D%B0%EC%9D%B4%ED%84%B0_EDA/10-7.png)

- 1554번, 1493번 영화의 상영 횟수가 많음을 파악하였음

**영화 코드별 수익 시각화**

*상영횟수가 많은 영화일수록 수익이 높은지를 파악해보고자 함*

```python
df6 = df.groupby('film_code')['total_sales'].sum().reset_index()
fig = px.bar(df6, x='film_code', y='total_sales', color='film_code', text='total_sales', title='영화 코드별 수익 시각화')
fig.show()
```

![Untitled](https://github.com/bjh0507/Study/blob/main/%EC%95%84%EB%A7%88%EC%A1%B4%EB%8D%B0%EC%9D%B4%ED%84%B0_EDA/10-8.png)

- 1554번 영화가 가장 높은 수익을 보이고, 1493번 영화의 수익은 1493번 영화보다 낮은 상영수를 보이는 몇몇의 영화보다 낮음을 파악
    - 상영횟수가 많다고 무조건 수익이 높은 것은 아님

**영화 코드별 상영횟수 대비 티켓 환불율 시각화**

```python
df7 = df.groupby('film_code')['tickets_out'].sum().reset_index()
df7['tickets_out_per'] = (df7['tickets_out'] / df5['show_time']).round(2)
fig = px.bar(df7, x='film_code', y='tickets_out_per', color='film_code', text='tickets_out_per', title='영화 코드별 상영 횟수 대비 티켓 환불 비율')
fig.show()
```

![Untitled](https://github.com/bjh0507/Study/blob/main/%EC%95%84%EB%A7%88%EC%A1%B4%EB%8D%B0%EC%9D%B4%ED%84%B0_EDA/10-9.png)

- 1482번, 1550번 영화의 티켓 환불율이 가장 높음을 파악할 수 있음

---

## 모델링

**데이터 전처리(스케일링)**

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  MinMaxScaler

scaler = MinMaxScaler()

df1 = df.drop(columns=['film_code', 'cinema_code', 'date', 'month', 'quarter', 'day'])
Y = df1['total_sales']
df1.drop(columns='total_sales', inplace=True)
train_x, val_x, train_y, val_y = train_test_split(df1, Y, test_size=0.3, random_state=77)
print(train_x.shape, val_x.shape)
print(train_y.shape, val_y.shape)
```

![Untitled](https://github.com/bjh0507/Study/blob/main/%EC%95%84%EB%A7%88%EC%A1%B4%EB%8D%B0%EC%9D%B4%ED%84%B0_EDA/10-10.png)

![Untitled](https://github.com/bjh0507/Study/blob/main/%EC%95%84%EB%A7%88%EC%A1%B4%EB%8D%B0%EC%9D%B4%ED%84%B0_EDA/10-11.png)

```python
train = pd.DataFrame(scaler.fit_transform(train_x))
train.columns = df1.columns
train
```

![Untitled](https://github.com/bjh0507/Study/blob/main/%EC%95%84%EB%A7%88%EC%A1%B4%EB%8D%B0%EC%9D%B4%ED%84%B0_EDA/10-12.png)

- 스케일링을 통해 컬럼별 표준화 수행

**타겟 변수**

```python
train_y = pd.DataFrame(train_y).reset_index()
train_y.drop(columns=['index'], inplace=True)
train_y
```

![Untitled](https://github.com/bjh0507/Study/blob/main/%EC%95%84%EB%A7%88%EC%A1%B4%EB%8D%B0%EC%9D%B4%ED%84%B0_EDA/10-13.png)

**K-fold를 이용한 LightGBM**

```python
from sklearn.model_selection import KFold

k_fold = KFold(n_splits=5, shuffle=True, random_state=77)

import lightgbm as lgb

model = lgb.LGBMRegressor(random_state=77, n_estimators=1000)
models = []
for train_idx, val_idx in k_fold.split(train) : 
  train_x = train.iloc[train_idx]
  y_train = train_y.iloc[train_idx]
  val_x = train.iloc[val_idx]
  val_y = train_y.iloc[val_idx]
  models.append(model.fit(train_x, y_train, eval_set=(val_x, val_y), early_stopping_rounds=100, verbose=100))
```

![Untitled](https://github.com/bjh0507/Study/blob/main/%EC%95%84%EB%A7%88%EC%A1%B4%EB%8D%B0%EC%9D%B4%ED%84%B0_EDA/10-14.png)

**모델 예측**

```python
pred = []
for model in models :
  pred.append(model.predict(val_x))

pd.DataFrame(pred)
```

![Untitled](https://github.com/bjh0507/Study/blob/main/%EC%95%84%EB%A7%88%EC%A1%B4%EB%8D%B0%EC%9D%B4%ED%84%B0_EDA/10-15.png)

- 5번 반복하여 모델 학습 수행

```python
import numpy as np
val_x['predict'] = np.mean(pred, axis=0).round(0)
val_x = pd.concat([val_x, val_y], axis=1)
val_x
```

![Untitled](https://github.com/bjh0507/Study/blob/main/%EC%95%84%EB%A7%88%EC%A1%B4%EB%8D%B0%EC%9D%B4%ED%84%B0_EDA/10-16.png)

- 예측값과 실제값을 비교해본 결과, 어느정도 비슷하게 예측을 수행했다는 것을 알 수 있음
