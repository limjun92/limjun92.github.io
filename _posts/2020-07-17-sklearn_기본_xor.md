---
title:  "sklearn_기본_xor"
excerpt: "AI_Algorithm  sklearn_기본_xor"
categories:
  - AI_개념
tags:
  - AI
  - AI_Algorithm
  - 정규 표현식
  - 머신러닝
last_modified_at: 2020-07-17
---

# 머신러닝 vs 딥러닝

|구분| Machine Learning| Deep Learning|
|---|---|---|
|훈련 데이터 크기| 작음| 큼|
|시스템 성능| 저 사양| 고 사양|
|feature 선택| 전문가 (사람) |알고리즘|
|feature 수| 많음 |적음|
|문제 해결 접근법| 문제를 분리 -> 각각 답을 얻음 -> 결과 통합| end-to-end (결과를 바로 얻음)|
|실행 시간| 짧음 |김|
|해석력 |해석 가능| 해석 어려움|


# scikit-learn

- 파이썬에 머신러닝 프레임워크 라이브러리
- 회귀, 분류, 군집, 차원축소, 특성공학, 전처리, 교차검증, 파이프라인 등 머신러닝에 필요한 기능 제공
- 학습을 위한 샘플 데이터 제공

# XOR 연산 학습해보기

## 일반
```python
from sklearn import svm

# 머신러닝에서 sklearn이 좋다

# xor 의 계산 결과 데이터
xor_input = [
    #p,q,r
    [0,0,0],
    [0,1,1],
    [1,0,1],
    [1,1,0]
]

xor_data = []
xor_label = []
for row in xor_input:
    p = row[0]
    q = row[1]
    r = row[2]
    xor_data.append([p,q])
    xor_label.append(r)
# 학습을 위해 데이터와 레이블 분리하기

model = svm.SVC()
        # 행과 열 , 일차원
model.fit(xor_data,xor_label)
# 데이터 학습시키기
        
pre = model.predict(xor_data)
print("예측데이터: ", xor_data)
print("예측 결과: ", pre)
# 데이터 예측하기

ok = 0; total = 0
for idx, answer in enumerate(xor_label):
    p = pre[idx]
    if p == answer:
        ok +=1
    total +=1
    
print("정답률: ", ok , "/", total, "=", ok/total)
# 결과 확인하기
```

예측데이터 :  [[0, 0], [0, 1], [1, 0], [1, 1]]  
예측  결과 :  [0 1 1 0]  
정답률: 4 / 4 = 1.0  

## pandas 라이브러리를 사용하여 코드 간략화

```python
import pandas as pd
from sklearn import svm, metrics

# XOR 연산
xor_input = [
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0]
]

# 학습을 위해 데이터와 레이블 분리하기 --- (※1)
xor_df = pd.DataFrame(xor_input)
xor_data  = xor_df[[0,1]] # 데이터
xor_label = xor_df[2]     # 레이블

# 데이터 학습과 예측하기 --- (※2)
model = svm.SVC()
model.fit(xor_data, xor_label)
pre = model.predict(xor_data)

# 정답률 구하기 --- (※3)
ac_score = metrics.accuracy_score(xor_label, pre)
print("정답률 =", ac_score)
```

정답률 = 1.0

## KNN 분류 모델을 이용
```python

import pandas as pd
from sklearn import svm, metrics
from sklearn.neighbors import KNeighborsClassifier # <- 모델추가

# XOR 연산
xor_input = [
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0]
]

# 학습을 위해 데이터와 레이블 분리하기 --- (※1)
xor_df = pd.DataFrame(xor_input)
xor_data  = xor_df[[0,1]] # 데이터
xor_label = xor_df[2]     # 레이블

# 데이터 학습과 예측하기 --- (※2)
clf = KNeighborsClassifier(n_neighbors=1)        # <- 모델변경
clf.fit(xor_data, xor_label)
pre = clf.predict(xor_data)

# 정답률 구하기 --- (※3)
ac_score = metrics.accuracy_score(xor_label, pre)
print("정답률 =", ac_score)
```
정답률 = 1.0
