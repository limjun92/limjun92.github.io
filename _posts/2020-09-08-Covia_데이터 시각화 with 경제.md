---
header:
  teaser: /assets/images/Covid_jun_files/Covid_jun_34_1.png
title:  "Covia_데이터 시각화 with 경제"
excerpt: "pandas, matplotlib"
toc: true
toc_sticky: true
categories:
  - pandas
tags:
  - pandas
  - matplotlib
last_modified_at: 2020-09-08
---

# 코로나가 경제에 미친영향
* 코로나 확진자의 통계정보를 간단하게 확인하고 코로나가 경제에 미친 영향을 벤처기업 대출금 데이터와 카드소비 데이터를 바탕으로 알아보자

## 코로나 통계 요약 정보

* 시각화 자료


```python
import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno
import folium
from folium.plugins import HeatMap
import seaborn as sns
from datetime import datetime
import numpy as np
```


```python
# Pandas 데이터 프레임에서 float을 소수점 2자리 까지 출력해 준다.
pd.set_option('display.float_format', lambda x: '%.2f' % x)
# 경고를 꺼준다.
import warnings  
warnings.filterwarnings('ignore')
plt.rc('font', family='NanumGothic')
```


```python
# 환자정보
PatientInfo = pd.read_csv(r'.\data\PatientInfo.csv')
# 환자이동경로
PatientRoute = pd.read_csv(r'.\data\PatientRoute.csv')
# 시간대별 확진자, 회복자, 사망자
time = pd.read_csv(r'.\data\time.csv')
# 시간대별 검색어
search_trend = pd.read_csv(r'.\data\SearchTrend.csv')
# 시간대별 나이감염
timeage = pd.read_csv(r'.\data\TimeAge.csv')
# 시간대별 성별감염
timegender = pd.read_csv(r'.\data\TimeGender.csv')
# 시간대별 지역감염
timeprovince = pd.read_csv(r'.\data\TimeProvince.csv')
# 시간대별 카드 사용처와 카드 사용횟수, 총 금액
card = pd.read_csv(r'.\data\card_20200717.csv')
# 시간대별 벤처기업 지원금
venture = pd.read_csv(r'.\data\Venture.csv',encoding='euc-kr')
```


```python
PatientInfo.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>patient_id</th>
      <th>global_num</th>
      <th>sex</th>
      <th>birth_year</th>
      <th>age</th>
      <th>country</th>
      <th>province</th>
      <th>city</th>
      <th>disease</th>
      <th>infection_case</th>
      <th>infection_order</th>
      <th>infected_by</th>
      <th>contact_number</th>
      <th>symptom_onset_date</th>
      <th>confirmed_date</th>
      <th>released_date</th>
      <th>deceased_date</th>
      <th>state</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1000000001</td>
      <td>2.00</td>
      <td>male</td>
      <td>1964</td>
      <td>50s</td>
      <td>Korea</td>
      <td>Seoul</td>
      <td>Gangseo-gu</td>
      <td>NaN</td>
      <td>overseas inflow</td>
      <td>1.00</td>
      <td>NaN</td>
      <td>75</td>
      <td>2020-01-22</td>
      <td>2020-01-23</td>
      <td>2020-02-05</td>
      <td>NaN</td>
      <td>released</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1000000002</td>
      <td>5.00</td>
      <td>male</td>
      <td>1987</td>
      <td>30s</td>
      <td>Korea</td>
      <td>Seoul</td>
      <td>Jungnang-gu</td>
      <td>NaN</td>
      <td>overseas inflow</td>
      <td>1.00</td>
      <td>NaN</td>
      <td>31</td>
      <td>NaN</td>
      <td>2020-01-30</td>
      <td>2020-03-02</td>
      <td>NaN</td>
      <td>released</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1000000003</td>
      <td>6.00</td>
      <td>male</td>
      <td>1964</td>
      <td>50s</td>
      <td>Korea</td>
      <td>Seoul</td>
      <td>Jongno-gu</td>
      <td>NaN</td>
      <td>contact with patient</td>
      <td>2.00</td>
      <td>2002000001</td>
      <td>17</td>
      <td>NaN</td>
      <td>2020-01-30</td>
      <td>2020-02-19</td>
      <td>NaN</td>
      <td>released</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1000000004</td>
      <td>7.00</td>
      <td>male</td>
      <td>1991</td>
      <td>20s</td>
      <td>Korea</td>
      <td>Seoul</td>
      <td>Mapo-gu</td>
      <td>NaN</td>
      <td>overseas inflow</td>
      <td>1.00</td>
      <td>NaN</td>
      <td>9</td>
      <td>2020-01-26</td>
      <td>2020-01-30</td>
      <td>2020-02-15</td>
      <td>NaN</td>
      <td>released</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1000000005</td>
      <td>9.00</td>
      <td>female</td>
      <td>1992</td>
      <td>20s</td>
      <td>Korea</td>
      <td>Seoul</td>
      <td>Seongbuk-gu</td>
      <td>NaN</td>
      <td>contact with patient</td>
      <td>2.00</td>
      <td>1000000002</td>
      <td>2</td>
      <td>NaN</td>
      <td>2020-01-31</td>
      <td>2020-02-24</td>
      <td>NaN</td>
      <td>released</td>
    </tr>
  </tbody>
</table>
</div>



* 결측값을 확인한다


```python
def mson_matrix(df,ax=None):
    msno.matrix(df,ax=ax)

figure, (ax1,ax2) = plt.subplots(nrows=2)
figure.set_size_inches(10,12)

mson_matrix(PatientInfo,ax1)
mson_matrix(PatientRoute,ax2)
```


![png](Covid_jun_files/Covid_jun_7_0.png)


* 의미 없는 값을 제거해준다


```python
idx_num = PatientInfo[(PatientInfo['age'] == '100s') | (PatientInfo['age'] == '30')].index
idx_num

PatientInfo.drop(idx_num, inplace = True)
```

* 현재까지 코로나 확진자들의 특징 요약


```python
def bar_chart(df, ax=None):
    df.plot(kind='bar', ax=ax)

figure, ((ax1,ax2,ax3)) = plt.subplots(nrows=1,ncols=3)
figure.set_size_inches(18,10)

sex = PatientInfo['sex'].value_counts()
age = PatientInfo['age'].value_counts().sort_index()
province = PatientInfo['province'].value_counts()

bar_chart(sex,ax1)
bar_chart(age,ax2)
bar_chart(province,ax3)

plt.show()
```


![png](/assets/images/Covid_jun_files/Covid_jun_11_0.png)


* 확진자들의 경로 확인


```python
for i in PatientRoute.index[:]:
    folium.Circle(
        location = PatientRoute.loc[i,['latitude','longitude']],
        radius = 200
    ).add_to(m)
m
```

* 나이대별 경로 변경횟수와 나이대별 확진자수의 관계


```python
info_route = pd.merge(PatientInfo,PatientRoute, on='patient_id')
age_route = info_route['age'].value_counts().sort_index()
plt.plot(age_route)
plt.plot(age)
```




    [<matplotlib.lines.Line2D at 0x244ae830490>]




![png](/assets/images/Covid_jun_files/Covid_jun_15_1.png)


## 시간의 흐름에 따른 시각화 자료

1. 테스트 횟수와 확진자 수 증가량
2. 확진자와 회복자
3. 검색량과 코로나 확진자
4. 나이별 확진자 추세
5. 나이별 사망자 추세
6. 성별 확진자 추세
7. 지역별 확진자 추세



```python
# 1. 테스트 횟수와 확진자 수 증가량
fig, ax = plt.subplots()

plt.rcParams['figure.figsize']=20,10
plt.xticks(rotation = - 90 )

sns.lineplot(x="date", y="test", data=time, ax=ax)
ax2 = ax.twinx()
sns.lineplot(x='date', y='confirmed', data=time, ax=ax2, color='r')
```




    <AxesSubplot:label='5e2640fd-6d2f-4eae-97a5-93e543c1669f', xlabel='date', ylabel='confirmed'>




![png](/assets/images/Covid_jun_files/Covid_jun_17_1.png)



```python
# 2. 확진자와 회복자
plt.rcParams['figure.figsize']=20,10
plt.xticks(rotation = - 90 )
sns.lineplot(x="date", y="confirmed", data=time)

sns.lineplot(x="date", y="released", data=time)
```




    <AxesSubplot:xlabel='date', ylabel='released'>




![png](/assets/images/Covid_jun_files/Covid_jun_18_1.png)



```python
# 3. 검색량과 코로나 확진자
trend_time = pd.merge(time,search_trend, on='date')
fig, ax = plt.subplots()

plt.rcParams['figure.figsize']=20,10
plt.xticks(rotation = - 90 )

sns.lineplot(x="date", y="coronavirus", data=trend_time, ax=ax)
ax2 = ax.twinx()
sns.lineplot(x='date', y='confirmed', data=trend_time, ax=ax2, color='r')
```




    <AxesSubplot:label='9b91c83a-805a-4481-9ff6-1362f83a7ecf', xlabel='date', ylabel='confirmed'>




![png](/assets/images/Covid_jun_files/Covid_jun_19_1.png)



```python
# 4. 나이별 확진자 추세

fig, ax = plt.subplots()

plt.rcParams['figure.figsize']=20,10
plt.xticks(rotation = - 90 )

ages = ['20s','30s','40s','50s','60s','70s','80s']

for age in ages:
    sns.lineplot(x="date", y="confirmed", data=timeage[timeage['age']== age], label=age)
```


![png](/assets/images/Covid_jun_files/Covid_jun_20_0.png)



```python
# 5. 나이별 사망자 추세

fig, ax = plt.subplots()

plt.rcParams['figure.figsize']=20,10
plt.xticks(rotation = - 90 )

ages = ['20s','30s','40s','50s','60s','70s','80s']

for age in ages:
    sns.lineplot(x="date", y="deceased", data=timeage[timeage['age']== age], label=age)
```


![png](/assets/images/Covid_jun_files/Covid_jun_21_0.png)



```python
# 6. 성별 확진자 추세

fig, ax = plt.subplots()

plt.rcParams['figure.figsize']=20,10
plt.xticks(rotation = - 90 )

sexs = ['female','male']

for sex in sexs:
    sns.lineplot(x="date", y="confirmed", data=timegender[timegender['sex']== sex], label=sex)
```


![png](/assets/images/Covid_jun_files/Covid_jun_22_0.png)



```python
# 7. 지역별 확진자 추세

fig, ax = plt.subplots()

plt.rcParams['figure.figsize']=20,10
plt.xticks(rotation = - 90 )

provinces = ['Seoul', 'Busan', 'Daegu', 'Incheon', 'Gwangju', 'Daejeon',
       'Ulsan', 'Sejong', 'Gyeonggi-do', 'Gangwon-do',
       'Chungcheongbuk-do', 'Chungcheongnam-do', 'Jeollabuk-do',
       'Jeollanam-do', 'Gyeongsangbuk-do', 'Gyeongsangnam-do', 'Jeju-do']

for province in provinces:
    sns.lineplot(x="date", y="confirmed", data=timeprovince[timeprovince['province']== province], label=province)
```


![png](/assets/images/Covid_jun_files/Covid_jun_23_0.png)


## 벤처기업 대출금 데이터 활용


```python
venture.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>일련번호</th>
      <th>지역</th>
      <th>업력</th>
      <th>대출년도</th>
      <th>대출월</th>
      <th>대출금액(백만원)</th>
      <th>업종</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>강원</td>
      <td>20년이상</td>
      <td>2020</td>
      <td>2</td>
      <td>200</td>
      <td>전자</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>강원</td>
      <td>20년미만</td>
      <td>2020</td>
      <td>2</td>
      <td>300</td>
      <td>화공</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>강원</td>
      <td>10년미만</td>
      <td>2020</td>
      <td>2</td>
      <td>200</td>
      <td>전기</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>강원</td>
      <td>15년미만</td>
      <td>2020</td>
      <td>2</td>
      <td>290</td>
      <td>화공</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>강원</td>
      <td>7년미만</td>
      <td>2020</td>
      <td>2</td>
      <td>300</td>
      <td>기타</td>
    </tr>
  </tbody>
</table>
</div>



### 월별 대출을 한 벤처기업의 수


```python
venture_month_count = venture['대출월'].value_counts().sort_index()
plt.rcParams['figure.figsize']=8,8
venture_month_count.plot(kind='bar')
```




    <AxesSubplot:>




![png](/assets/images/Covid_jun_files/Covid_jun_27_1.png)


### 월별 벤처기업 대출금의 총금액


```python
total = venture.pivot_table(values='대출금액'+'('+'백만원'+')',index ='대출월',aggfunc=sum)
total.plot(kind='bar')
```




    <AxesSubplot:xlabel='대출월'>




![png](/assets/images/Covid_jun_files/Covid_jun_29_1.png)


## 카드 소비 데이터


```python
sales = card.groupby(['mrhst_induty_cl_nm','receipt_dttm']).sum()['salamt'].reset_index()
sales
sales.receipt_dttm = [datetime.strptime(str(x),'%Y%m%d').month for x in sales.receipt_dttm]
sales = sales.groupby(['mrhst_induty_cl_nm','receipt_dttm']).sum()['salamt'].reset_index()
#sales = sales[sales.receipt_dttm !=6]

total = sales.pivot_table(values=sales[sales['receipt_dttm']==2], index='receipt_dttm',columns='mrhst_induty_cl_nm')
risk2 = total.loc[2] - total.loc[1]
risk3 = total.loc[3] - total.loc[2]
mean_sales = sales.groupby('mrhst_induty_cl_nm').mean()
```


```python
mean_sales.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>receipt_dttm</th>
      <th>salamt</th>
    </tr>
    <tr>
      <th>mrhst_induty_cl_nm</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1급 호텔</th>
      <td>3.50</td>
      <td>927609955.67</td>
    </tr>
    <tr>
      <th>2급 호텔</th>
      <td>3.50</td>
      <td>1006486824.17</td>
    </tr>
    <tr>
      <th>CATV</th>
      <td>3.50</td>
      <td>114058693.50</td>
    </tr>
    <tr>
      <th>CATV홈쇼핑</th>
      <td>3.50</td>
      <td>34443030830.67</td>
    </tr>
    <tr>
      <th>L P G</th>
      <td>3.50</td>
      <td>2424949584.00</td>
    </tr>
  </tbody>
</table>
</div>




```python
risk_data = pd.DataFrame({'risk':(risk2 + risk3)/mean_sales['salamt']})
risk_data.values
risk = np.ravel(risk_data.values, order='C')
risk_data = pd.DataFrame({'index':sales['mrhst_induty_cl_nm'].unique(),'risk':risk})
risk_data = risk_data.dropna(axis=0)
risk_data
risk_data_remove =  risk_data.drop(risk_data[(risk_data.risk < 1) & (risk_data.risk > -1)].index)
```


```python
plt.rcParams['figure.figsize']=20,10
plt.xticks(rotation = - 90 )
plt.grid(True)
plt.bar(risk_data_remove['index'],risk_data_remove['risk'])
```




    <BarContainer object of 35 artists>




![png](/assets/images/Covid_jun_files/Covid_jun_34_1.png)


* 이동과 관련된 '관광여행'과 '항공사'의 카드 사용량이 확실히 준것을 확인할 수 있다
