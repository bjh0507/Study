# 1&2주차-Amazon 구매 데이터

# 데이터

아마존 상품 데이터 orders_data.xlsx

# 필요한 라이브러리

```jsx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import plotly.express as px
warnings.simplefilter('ignore')

from wordcloud import WordCloud, STOPWORDS
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.corpus import stopwords

%matplotlib inline
```

# 데이터 전처리

```jsx
df.head(10)
```

![Untitled](https://github.com/bjh0507/Study/blob/main/%EC%95%84%EB%A7%88%EC%A1%B4%EB%8D%B0%EC%9D%B4%ED%84%B0_EDA/1-1.png)

### ship_city & ship_state 변수의 대문자 변환 및 문자열 ‘,’ 제거 작업

```jsx
places = ['ship_city', 'ship_state']
for i in places:
  df[i] = df[i].apply(lambda x:x.upper())

df['ship_city'] = df['ship_city'].apply(lambda x: x.replace(',', ''))
df['ship_state'] = df['ship_state'].apply(lambda x: x.replace(',', ''))
```

![Untitled](https://github.com/bjh0507/Study/blob/main/%EC%95%84%EB%A7%88%EC%A1%B4%EB%8D%B0%EC%9D%B4%ED%84%B0_EDA/1-2.png)

→ apply함수를 사용하여 lambda를 통해 ship_city와 ship_state 변수에 ‘,’ 대신 공백으로 대체한 결과 문자열이 사라지고 대문자 변환이 잘된것을 볼 수 있음

### 결측치 확인 및 처리

```jsx
df.isnull().sum()
```

![Untitled](https://github.com/bjh0507/Study/blob/main/%EC%95%84%EB%A7%88%EC%A1%B4%EB%8D%B0%EC%9D%B4%ED%84%B0_EDA/1-3.png)

→ item_total 18개, shipping_fee 26개, cod 124개의 결측치가 있다는 것을 파악

```jsx
df['shipping_fee'].fillna(df['shipping_fee'].mode()[0], inplace=True)
df['item_total'].fillna(df['item_total'].mode()[0], inplace=True)
df['cod'].fillna('online', inplace=True)
```

![Untitled](https://github.com/bjh0507/Study/blob/main/%EC%95%84%EB%A7%88%EC%A1%B4%EB%8D%B0%EC%9D%B4%ED%84%B0_EDA/1-4.png)

- shipping_fee & item_total 값은 mode함수를 사용해 해당 변수 중 최빈값을 해당하는 값을 결측치 대체
- cod 변수는 배송 후 결제 이외로 구매할 수 있는 방법은 online밖에 없기에 결측치를 online으로 대체

### 루피기호 제거

```jsx
amount = ['item_total', 'shipping_fee']
for i in amount :
  df[i] = df[i].apply(lambda x : x[1:])
```

→ 루피 기호를 제거하기 위해 apply함수와 lambda를 활용해 0번째 인덱스에 존재하는 루피를 제외한 모든 값을 가져와야 하므로 [1:] 슬라이싱 수행

# 데이터 시각화

- 결제 방식이 구매결정여부에 미치는 영향을 파악하고자 함

### 데이터 타입 변경

```jsx
i = 'int64'
f = 'float64'
df = df.astype({'item_total' : f, 'shipping_fee' : f, 'quantity' : i})
```

### 필요한 변수 추출

- reset_index()함수 작동방법

```jsx
df[['cod', 'order_status', 'item_total']].groupby(['cod', 'order_status']).sum()
```

![Untitled](https://github.com/bjh0507/Study/blob/main/%EC%95%84%EB%A7%88%EC%A1%B4%EB%8D%B0%EC%9D%B4%ED%84%B0_EDA/1-5.png)

→ reset_index함수를 부여하지 않으면 cod변수가 인덱스로 들어감

```jsx
df[['cod', 'order_status', 'item_total']].groupby(['cod', 'order_status']).sum().reset_index()
```

![Untitled](https://github.com/bjh0507/Study/blob/main/%EC%95%84%EB%A7%88%EC%A1%B4%EB%8D%B0%EC%9D%B4%ED%84%B0_EDA/1-6.png)

→ reset_index 적용시 cod변수가 인덱스에서 빠져나와 독립적인 변수 형태로 보여줌

### plotly라이브러리의 히스토그램

```jsx
fig = px.histogram(df, x='cod', y='item_total', color='order_status', barmode='group', height=450)
fig.show()
```

![Untitled](https://github.com/bjh0507/Study/blob/main/%EC%95%84%EB%A7%88%EC%A1%B4%EB%8D%B0%EC%9D%B4%ED%84%B0_EDA/1-7.png)

→ 각 범주별 그룹 옵션을 주고, order_status 별 색상을 부여해 히스토그램을 그림

해석 : 온라인을 통해 물건을 구매하는 경우가 많은 것을 알 수 있음

### Date에 따른 총 판매 가격

```jsx
df['year'] = pd.DatetimeIndex(df['order_date']).year #년도만 추출
df['date'] = pd.DatetimeIndex(df['order_date']).date #날짜 추출
df['time'] = pd.DatetimeIndex(df['order_date']).time #시간 추출
df['month_name'] = pd.DatetimeIndex(df['date']).month_name() #월 이름 변수
df['day_name'] = pd.DatetimeIndex(df['date']).day_name() #요일 변수
```

→ 날짜 포맷을 통해 년도, 월, 요일, 시간을 추출해 새로운 변수로 지정

```jsx
df['sku'] = df['sku'].apply(lambda x: x[4:])
df.head(3)
```

→ sku변수 값에 중복적으로 적힌 ‘sku:’ 문자열 제거

```jsx
df_sales = df[['date', 'item_total']].groupby('date').sum().reset_index() 
df_sales.head(3)
```

![Untitled](https://github.com/bjh0507/Study/blob/main/%EC%95%84%EB%A7%88%EC%A1%B4%EB%8D%B0%EC%9D%B4%ED%84%B0_EDA/1-8.png)

→ 필요한 변수인 date, item_total을 가져와 groupby함수를 적용해 표 작성

### plotly line함수 적용

```jsx
fig = px.line(df_sales, x='date', y='item_total', title='Sales ocer the period')
fig.show()
```

![Untitled](https://github.com/bjh0507/Study/blob/main/%EC%95%84%EB%A7%88%EC%A1%B4%EB%8D%B0%EC%9D%B4%ED%84%B0_EDA/1-9.png)

→ 12월 21일 아이템 판매량이 가장 높고, 대체적으로 12월달의 판매액이 큰 것을 알 수 있음

## 월별/요일별 판매량

```jsx
fig_box = px.bar(df_month, x='month_name', y='item_total', color='day_name', title='Month-wise Sales')
fig_box.show()

```

![Untitled](https://github.com/bjh0507/Study/blob/main/%EC%95%84%EB%A7%88%EC%A1%B4%EB%8D%B0%EC%9D%B4%ED%84%B0_EDA/1-10.png)

→ 판매량이 가장 높은 12월은 화요일에 매출액이 가장 큰 것을 파악할 수 있고, 대체적으로 목요일 & 금요일에 매출액이 큰 것을 알 수 있음

### 배송지역 관련 선버스트 차트

```jsx
df_sunburst = df[['ship_city', 'ship_state', 'item_total']].groupby(['ship_city', 'ship_state']).sum().reset_index()
df_sunburst.head(3)
df_sunburst.sort_values(by='item_total', ascending=False, inplace=True)
```

→ 필요한 변수를 추출하여 그룹화를 진행하고, 선버스트 차트는 데이터를 내림차순으로 정렬하여 수행하는 것이 바람직함

```jsx
fig = px.sunburst(data_frame = df_sunburst, path=['ship_state', 'ship_city'], values='item_total', title='Region-wise orders')
fig.show()
```

![Untitled](https://github.com/bjh0507/Study/blob/main/%EC%95%84%EB%A7%88%EC%A1%B4%EB%8D%B0%EC%9D%B4%ED%84%B0_EDA/1-11.png)

→ 선버스트 차트는 안에 위치한 원일수록 높은 수준의 계층을 의미함

### 상위 5개 지역 주문량

```jsx
top_city = df.groupby('ship_city').size().reset_index().rename(columns={0:'Total'}).sort_values('Total', ascending=False).head(5)
```

→ 변수 추출 필요 없이 ship_city(지역) 변수를 기준으로 그룹화를 진행해 내림차순으로 정렬하여 head()함수로 상위 5개를 추출해 top_city라는 변수에 담음

- size()함수는 ship_city의 변수가 수치가 아닌 범주형이므로 범주형의 count개수를 출력하기 위해 사용
- 각 범주별 count한 개수가 ‘0’이라는 변수의 이름에 담겨져 있음→rename함수를 통해 0을 Total로 변수 이름 재정의

### plotly 파이차트

```jsx
fig = px.pie(top_city, values='Total', names='ship_city', color_discrete_sequence=px.colors.sequential.RdBu, title='Top 5 ordering cities')
fig.show()
```

![Untitled](https://github.com/bjh0507/Study/blob/main/%EC%95%84%EB%A7%88%EC%A1%B4%EB%8D%B0%EC%9D%B4%ED%84%B0_EDA/1-12.png)

→ 상위 5개 지역을 color_discrete_sequence 옵션을 사용해 레드계열의 색으로 각 범주 별 고유색을 지정해 차이를 분명하게 나타내고자 함

### 상위 5개 item

```jsx
top_products = df.groupby('sku').size().reset_index().rename(columns={0:'Total'}).sort_values('Total', ascending=False).head(5)
top_products.head()
```

```jsx
fig = px.pie(top_products, values='Total', names='sku', color_discrete_sequence=px.colors.sequential.BuGn_r, title='Top 5 Products')
fig.show()
```

![Untitled](https://github.com/bjh0507/Study/blob/main/%EC%95%84%EB%A7%88%EC%A1%B4%EB%8D%B0%EC%9D%B4%ED%84%B0_EDA/1-13.png)

→ 상위 5개 item 품목을 초록색 계열로 고유색을 부여해 파이차트를 그림
