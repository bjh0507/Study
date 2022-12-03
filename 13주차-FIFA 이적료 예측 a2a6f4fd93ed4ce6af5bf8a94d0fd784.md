# 13주차-FIFA 이적료 예측

### 라이브러리 업로드

```
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
```

### 데이터 업로드

```
train = pd.read_csv('/content/FIFA_train.csv')
test = pd.read_csv('/content/FIFA_test.csv')
train
```

![Untitled](https://github.com/bjh0507/Study/blob/main/%EC%95%84%EB%A7%88%EC%A1%B4%EB%8D%B0%EC%9D%B4%ED%84%B0_EDA/7-1.png)

```
df = train.copy()
df
```

시각화를 위해 train 데이터 복제본이 df 생성

### 나이대별 평균 이적료 시각화

```
df.age.value_counts()
```

![Untitled](https://github.com/bjh0507/Study/blob/main/%EC%95%84%EB%A7%88%EC%A1%B4%EB%8D%B0%EC%9D%B4%ED%84%B0_EDA/7-2.png)

```
#나이대별 평균 이적료

df['age_value'] = ""
df.loc[(df['age'] >= 10) & (df['age'] < 20) , 'age_value'] = '10대'
df.loc[(df['age'] >= 20) & (df['age'] < 30) , 'age_value'] = '20대'
df.loc[(df['age'] >= 30) & (df['age'] < 40) , 'age_value'] = '30대'
df.loc[(df['age'] >= 40) & (df['age'] < 50) , 'age_value'] = '40대'
df
```

![Untitled](https://github.com/bjh0507/Study/blob/main/%EC%95%84%EB%A7%88%EC%A1%B4%EB%8D%B0%EC%9D%B4%ED%84%B0_EDA/7-3.png)

```
df_age = pd.DataFrame(df.groupby('age_value')['value'].mean().round(2).sort_values(ascending=False))
fig = px.bar(df_age, x=df_age.index, y='value', color=df_age.index, title='연령대별 평균 이적료')
fig.show()
```

![Untitled](https://github.com/bjh0507/Study/blob/main/%EC%95%84%EB%A7%88%EC%A1%B4%EB%8D%B0%EC%9D%B4%ED%84%B0_EDA/7-4.png)

- 20, 30대 선수들의 평균 이적료가 높음을 알 수 있음

<aside>
💡 추가적으로 원래의 age변수로 나이별 평균 이적료도 시각화 해보기

</aside>

### 국적에 따른 이적료에 차이가 있는지 시각화

```
df_country = pd.DataFrame(df.groupby('continent')['value'].mean().round(2).sort_values(ascending=False))
df_country
```

![Untitled](https://github.com/bjh0507/Study/blob/main/%EC%95%84%EB%A7%88%EC%A1%B4%EB%8D%B0%EC%9D%B4%ED%84%B0_EDA/7-5.png)

```
fig = px.bar(df_country, x=df_country.index, y='value', color=df_country.index, title='국적별 이적료 시각화')
fig.show()
```

![Untitled](https://github.com/bjh0507/Study/blob/main/%EC%95%84%EB%A7%88%EC%A1%B4%EB%8D%B0%EC%9D%B4%ED%84%B0_EDA/7-6.png)

- 미국, 아프리카, 유럽쪽 선수들의 이적료가 다른 국적보다 높음을 알 수 있음
    - 국적에 따라 피지컬의 차이가 존재 → 추후 머신러닝 모델을 돌릴 때 피쳐로 활용 가능
    

### 왼발/오른발에 따른 이적료 시각화

```
df_foot = pd.DataFrame(df.groupby('prefer_foot')['value'].mean().round(3).sort_values(ascending=False))
fig = px.bar(df_foot, x=df_foot.index, y='value', color=df_foot.index)
fig.show()
```

![Untitled](https://github.com/bjh0507/Study/blob/main/%EC%95%84%EB%A7%88%EC%A1%B4%EB%8D%B0%EC%9D%B4%ED%84%B0_EDA/7-7.png)

- 왼발/오른발에 따른 평균 이적료 차이가 거의 없음 → 이적료에 중요한 영향을 미치는 변수가 아님

<aside>
💡 기존의 포지션 변수를 토대로 포지션별 선수 수 먼저 시각화 해보기 
→ 어떤 포지션의 선수들이 많은지 파악하고, 그에 따른 이적료가 높은 확인

</aside>

### 포지션별 평균 이적료

```
df_position = pd.DataFrame(df.groupby('position').mean().round(2))
fig = px.pie(df_position, names=df_position.index, values='value', color=df_position.index, title='포지션별 평균 이적료')
fig.show()
```

![Untitled](https://github.com/bjh0507/Study/blob/main/%EC%95%84%EB%A7%88%EC%A1%B4%EB%8D%B0%EC%9D%B4%ED%84%B0_EDA/7-8.png)

- 스트라이커와 미드필더의 포지션에서 높은 이적료를 보임
    - 또한, 포지션별 이적료 차이가 존재 → 의미 있는 변수

---

## 이적료 예측 모델링

### 데이터 전처리

```
Y = train['value']
train = train.drop(columns=['id', 'name', 'contract_until', 'prefer_foot', 'value', 'continent'])
```

```
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
train['position'] = encoder.fit_transform(train['position'])
```

```
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
train = scaler.fit_transform(train)
train
```

- train 데이터셋 전처리 → label encoder로 범주형을 수치형으로 변환, minmax스케일러 사용

```
test = test.drop(columns=['id', 'name', 'continent', 'prefer_foot', 'contract_until'])
test.position = encoder.fit_transform(test['position'])

test = scaler.transform(test)
test
```

- test데이터 셋도 train과 동일하게 전처리 수행
    - test 데이터 셋은 스케일링을 수행할 때 fit은 하면 안됨! transform만!

## 랜던포레스트-회귀 예측

```
from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(train, Y, test_size = 0.3, random_state=777) # 굳이 나눌 필요 없음 -> 그냥 타겟변수 자체를 학습 y로 사용하면 됨
print(train_x.shape, test_x.shape)
print(train_y.shape, test_y.shape)
```

```
#랜덤 회귀 예측
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
#model_rf = RandomForestRegressor(n_estimator=10) n_estimators : 생성할 트리 개수

list=[10,20,30]

for i in list:
  model_rf = RandomForestRegressor(n_estimators=i)
  model_rf.fit(train_x, train_y)
  y_pred = model_rf.predict(test_x)
  print('MAE score:', mean_absolute_error(test_y, y_pred))
```

![Untitled](https://github.com/bjh0507/Study/blob/main/%EC%95%84%EB%A7%88%EC%A1%B4%EB%8D%B0%EC%9D%B4%ED%84%B0_EDA/7-9.png)

- mae값이 낮으면 좋음
    - n_estimator(생성 트리 개수)가 30일 때 가장 성능이 좋음을 알 수 있음

## test데이터로 예측 수행

```
model_rf = RandomForestRegressor(n_estimators=30)
model_rf.fit(train_x, train_y)
prediction = model_rf.predict(test)
sample_submission = pd.read_csv('/content/submission.csv')
sample_submission['value'] = prediction.round(2)
sample_submission
```

![Untitled](https://github.com/bjh0507/Study/blob/main/%EC%95%84%EB%A7%88%EC%A1%B4%EB%8D%B0%EC%9D%B4%ED%84%B0_EDA/7-10.png)

### 선수의 국적을 나타내는 continent를 포함시켜 다시 수행

```
Y = train['value']
train = train.drop(columns=['id', 'name', 'contract_until', 'prefer_foot', 'value'])
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
train['position'] = encoder.fit_transform(train['position'])
train['continent'] = encoder.fit_transform(train['continent'])
train
```

```
train = scaler.fit_transform(train)
```

```
train_x, val_x, train_y, val_y = train_test_split(train, Y, test_size=0.3, random_state=777)
print(train_x.shape, train_y.shape)
print(val_x.shape, val_y.shape)
```

```
list = [10,20,30]

for i in list :
  model_rf = RandomForestRegressor(n_estimators=i)
  model_rf.fit(train_x, train_y)
  y_pred = model_rf.predict(val_x)
  print('MAE score:', mean_absolute_error(val_y, y_pred))
```

![Untitled](https://github.com/bjh0507/Study/blob/main/%EC%95%84%EB%A7%88%EC%A1%B4%EB%8D%B0%EC%9D%B4%ED%84%B0_EDA/7-11.png)

- 여전히 30일 때 가장 좋은 mae값을 보이나, 변수 추가 후 30일때의 mae값이 증가함을 알 수 있음
    - 변수의 중요도가 낮기에 굳이 포함시켜 모델을 돌릴 필요 없음

<aside>
💡 get_dummies VS LabelEncoder

- 독립적인 범주로 있을 때 get_dummies를 활용해 범주형을 수치화 시키는 것이 좋음
    - 독립적인 범주를 LabelEncoder로 하나의 변수 안에 수치화 시켜버리면 상관분석을 수행할 때 각 독립적인 범주의 영향을 제대로 반영할 수 없어서!
- 범주 간 순위가 존재할 때는 LabelEncoder가 더욱 적합함
</aside>
