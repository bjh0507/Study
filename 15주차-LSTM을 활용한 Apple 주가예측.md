# 15주차-LSTM을 활용한 Apple 주가예측

데이터(제공 : 캐글)

[APPL_DATA.csv](https://github.com/bjh0507/Study/blob/main/%EC%95%84%EB%A7%88%EC%A1%B4%EB%8D%B0%EC%9D%B4%ED%84%B0_EDA/APPL_DATA.csv)

```jsx
import pandas as pd

df = pd.read_csv('/content/APPL_DATA.csv')
df
```

![Untitled](https://github.com/bjh0507/Study/blob/main/%EC%95%84%EB%A7%88%EC%A1%B4%EB%8D%B0%EC%9D%B4%ED%84%B0_EDA/9-1.png)

데이터 확인

```jsx
df.info()
```

![Untitled](https://github.com/bjh0507/Study/blob/main/%EC%95%84%EB%A7%88%EC%A1%B4%EB%8D%B0%EC%9D%B4%ED%84%B0_EDA/9-2.png)

```jsx
df.isnull().sum()
```

![Untitled](https://github.com/bjh0507/Study/blob/main/%EC%95%84%EB%A7%88%EC%A1%B4%EB%8D%B0%EC%9D%B4%ED%84%B0_EDA/9-3.png)

```jsx
df['Date'] = pd.to_datetime(df['Date'])
df.info()
```

![Untitled](https://github.com/bjh0507/Study/blob/main/%EC%95%84%EB%A7%88%EC%A1%B4%EB%8D%B0%EC%9D%B4%ED%84%B0_EDA/9-4.png)

- 날짜 변수는 datetime으로 변환

**종가 기준 애플 주식 시각화**

```jsx
import plotly.express as px

fig = px.line(df, x='Date', y='Close', title='애플 주식 시각화(종가기준)')
fig.show()
```

![Untitled](https://github.com/bjh0507/Study/blob/main/%EC%95%84%EB%A7%88%EC%A1%B4%EB%8D%B0%EC%9D%B4%ED%84%B0_EDA/9-5.png)

- 2022년 1월달에 가장 높은 주가를 보임
- 2022년 1월까지 지속적으로 상승세를 보이다가 그 뒤부터 주가 상승과 하락의 변동성이 커짐을 파악할 수 있음

**데이터 정규화**

```
from sklearn.preprocessing import MinMaxScaler
x = df[['High', 'Low', 'Open', 'Close']]
scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(x))
X.columns = x.columns
X
```

![Untitled](https://github.com/bjh0507/Study/blob/main/%EC%95%84%EB%A7%88%EC%A1%B4%EB%8D%B0%EC%9D%B4%ED%84%B0_EDA/9-6.png)

- 모델 돌리기 전, 스케일링을 통해 변수의 범위를 맞춰 정규화를 시켜주면 모델의 성능을 높일 수 있음
- MinMaxScaler사용(0-1사이의 값으로 변환)

**train & val 셋 분할**

```
TEST_SIZE = 200
window_size = 20

train = X[:-TEST_SIZE]
test = X[-TEST_SIZE:]
```

![train데이터](https://github.com/bjh0507/Study/blob/main/%EC%95%84%EB%A7%88%EC%A1%B4%EB%8D%B0%EC%9D%B4%ED%84%B0_EDA/9-7.png)

train데이터

![test데이터](https://github.com/bjh0507/Study/blob/main/%EC%95%84%EB%A7%88%EC%A1%B4%EB%8D%B0%EC%9D%B4%ED%84%B0_EDA/9-8.png)

test데이터

```python
import numpy as np

def make_dataset(data, label, window_size = 20) :
  feature_list = []
  label_list = []
  for i in range(len(data) - window_size) :
    feature_list.append(np.array(data.iloc[i:i+window_size])) #train 셋에서 i번째 부터 I+20번째 행까지 추출해 하나의 배열에 담아 빈 리스트에 추가
    label_list.append(np.array(label.iloc[i+window_size]))
  return np.array(feature_list), np.array(label_list)
```

- input데이터를 만들기 위한 과정 : 데이터 → 행렬 변환

```jsx
from sklearn.model_selection import train_test_split

train_feature = train[['High', 'Low', 'Open']]
train_label = train['Close']

train_feature, train_label = make_dataset(train_feature, train_label, window_size=20)

x_train, x_val, y_train, y_val = train_test_split(train_feature, train_label, test_size = 0.3, random_state=3)
print(x_train.shape, x_val.shape)
```

![Untitled](https://github.com/bjh0507/Study/blob/main/%EC%95%84%EB%A7%88%EC%A1%B4%EB%8D%B0%EC%9D%B4%ED%84%B0_EDA/9-9.png)

- train 데이터 : 20X3 행렬이 7240개 존재
- val 데이터 : 20X3 행렬이 3104개 존재

**test 데이터 셋 전처리**

- train데이터와 동일하게 전처리

```python
test_feature = test[['High', 'Low', 'Open']]
test_label = test['Close']

test_feature.shape, test_label.shape
```

![Untitled](https://github.com/bjh0507/Study/blob/main/%EC%95%84%EB%A7%88%EC%A1%B4%EB%8D%B0%EC%9D%B4%ED%84%B0_EDA/9-10.png)

```python
test_feature, test_label = make_dataset(test_feature, test_label, 20)
test_feature.shape, test_label.shape
```

**모델 학습**

```python
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import LSTM

model = Sequential()
model.add(LSTM(16,
               input_shape=(train_feature.shape[1], train_feature.shape[2]), #input_shape으로 train_feature에 1번 인덱스와 2번 인덱스 값을 가져와 (20,3) 행렬을 input으로 집어 넣겠다는 의미
               activation='relu',
               return_sequences = False)
          )
model.add(Dense(1))
```

- train_feature.shape[1] : (7240, 20, 3)에서 20을 가져오겠다는 뜻
- train_feature.shape[2] : (7240, 20, 3)에서 3을 가져오겠다는 뜻

→ 즉, input shape 형태가 (20,3)형태의 행렬을 의미

```python
import os
model.compile(loss='mean_squared_error', optimizer='adam') #LSTM에서는 옵티마이저를 adam을 많이 씀
early_stop = EarlyStopping(monitor='val_loss', patience=5) #patience 옵션 : 

model_path = 'model'
filename = os.path.join(model_path, 'tmp_checkpoint.h5')
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

history = model.fit(x_train, y_train,
                    epochs=200,
                    batch_size=25,
                    validation_data=(x_val, y_val),
                    callbacks=[early_stop, checkpoint])
```

![Untitled](https://github.com/bjh0507/Study/blob/main/%EC%95%84%EB%A7%88%EC%A1%B4%EB%8D%B0%EC%9D%B4%ED%84%B0_EDA/9-11.png)

- LSTM에서 옵티마이저로 주로 adam을 많이 씀
- patience=5 옵션 : 검증 데이터 셋의 오차가 5번 이상 낮아지지 않는 경우 학습 종료를 의미
- verbose 옵션 : 0은 상세한 정보 출력X, 1은 자세한 정보들을 출력하라는 의미

**모델 예측**

```python
model.load_weights(filename)
pred = model.predict(test_feature)

pred
```

![Untitled](https://github.com/bjh0507/Study/blob/main/%EC%95%84%EB%A7%88%EC%A1%B4%EB%8D%B0%EC%9D%B4%ED%84%B0_EDA/9-12.png)

**예측 시각화**

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12,9))
plt.plot(test_label, label = 'actual')
plt.plot(pred, label = 'prediction')
plt.legend()
plt.show()
```

![Untitled](https://github.com/bjh0507/Study/blob/main/%EC%95%84%EB%A7%88%EC%A1%B4%EB%8D%B0%EC%9D%B4%ED%84%B0_EDA/9-13.png)

- 실제 주가와 비슷하게 예측한 것을 알 수 있음
