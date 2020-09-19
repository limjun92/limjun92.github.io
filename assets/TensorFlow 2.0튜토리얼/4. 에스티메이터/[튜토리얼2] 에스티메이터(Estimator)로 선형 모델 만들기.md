
# [튜토리얼2] 에스티메이터(Estimator)로 선형 모델 만들기

이번 튜토리얼에서는 `tf.estimator` API를 사용하여 로지스틱 회귀 모델(logistic regression model)을 훈련합니다. 

이 모델은 다른 더 복잡한 알고리즘의 기초로 사용할 수 있습니다.


```python
from __future__ import absolute_import, division, print_function, unicode_literals

import warnings
warnings.simplefilter('ignore')

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from IPython.display import clear_output
from six.moves import urllib

import tensorflow as tf
```

# 목차
1. 타이타닉 데이터셋 불러오기
2. 데이터 탐색하기
3. 모델을 위한 피쳐 공학(feture engineering)
    - 3.1 기본 피쳐 열(feature column)
    - 3.2 도출된 피쳐 열(feature column)

## 1. 타이타닉 데이터셋 불러오기
타이타닉 데이터셋을 사용할 것입니다. 성별, 나이, 클래스, 기타 등 주어진 정보를 활용하여 승객이 살아남을 것인지 예측하는 것을 목표로 합니다.


```python
# Pandas를 사용해 데이터셋 불러오기.
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')

# 레이블(살아남았는지 유무)
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')
```

## 2. 데이터 탐색하기

데이터셋은 다음의 특성을 가집니다


```python
dftrain.head()
```

학습에 사용할 피쳐들은 다음과 같습니다:

<table>
  <tr>
    <th>Feature Name</th>
    <th>Description</th>
  </tr>
  <tr>
    <td>sex</td>
    <td>승객의 성별</td>
  </tr>
  <tr>
    <td>age</td>
    <td>승객의 나이</td>
  </tr>
    <tr>
    <td>n_siblings_spouses</td>
    <td>함께 탑승한 형제와 배우자의 수</td>
  </tr>
    <tr>
    <td>parch</td>
    <td>함께 탑승한 부모, 아이의 수</td>
  </tr>
    <tr>
    <td>fare</td>
    <td>탑승료</td>
  </tr>
    <tr>
    <td>class</td>
    <td>티켓의 등급</td>
  </tr>
    <tr>
    <td>deck</td>
    <td>탑승한 갑판</td>
  </tr>
    <tr>
    <td>embark_town</td>
    <td>탑승 항구</td>
  </tr>
    <tr>
    <td>alone</td>
    <td>혼자인지에 대한 여부</td>
  </tr>
</table>


```python
dftrain.describe()
```

훈련 셋(dftrain)은 627개의 샘플로 평가 셋(dfeval)은 264개의 샘플로 구성되어 있습니다.


```python
dftrain.shape[0], dfeval.shape[0]
```

데이터 시각화를 통해 조금 더 자세히 알아보도록 하겠습니다.

<대부분의 승객은 **20대와 30대** 입니다.>


```python
dftrain.age.hist(bins=20)
```

<남자 승객이 여자 승객보다 **대략 2배** 많습니다.>


```python
dftrain.sex.value_counts().plot(kind='barh')
```

<대부분의 승객은 **"3등석"** 입니다.>


```python
dftrain['class'].value_counts().plot(kind='barh')
```

<여자는 남자보다 **살아남을 확률**이 훨씬 높습니다. 이는 확실히 모델에 유용한 피쳐입니다.>


```python
pd.concat([dftrain, y_train], axis=1).groupby('sex').survived.mean().plot(kind='barh').set_xlabel('% survive')
```

## 3. 모델을 위한 피쳐 엔지니어링(feature engineering)
에스티메이터는 **피쳐 열(feature columns)** 이라는 시스템을 사용하여 모델이 각각의 입력 피쳐를 어떻게 해석할지 설명합니다. 예를 들어, 에스티메이터가 숫자 입력 벡터를 요구하면, **피쳐 열**은 모델이 어떻게 각 피쳐를 변환해야하는지 설명합니다.

효과적인 모델 학습에서는 적절한 피쳐 열을 고르고 다듬는 것이 핵심입니다. 하나의 피쳐 열은 피쳐 딕셔너리(dict)의 원본 입력으로 만들어진 열(기본 피쳐 열)이거나 하나 이상의 기본 열(얻어진 피쳐 열)에 정의된 변환을 이용하여 새로 생성된 열입니다.

선형 에스티메이터는 수치형, 범주형 피쳐를 모두 사용할 수 있습니다. 피쳐 열은 모든 텐서플로우 에스티메이터와 함께 작동하고 목적은 모델링에 사용되는 피쳐들을 정의하는 것입니다. 또한 원-핫-인코딩(one-hot-encoding), 정규화(normalization), 버킷화(bucketization)와 같은 피쳐 엔지니어링 방법을 지원합니다.

### 3.1 기본 피쳐 열


```python
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                       'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = dftrain[feature_name].unique()
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))
```

`input_function`은 입력 파이프라인을 스트리밍으로 공급하는 `tf.data.Dataset`으로 데이터를 변환하는 방법을 명시합니다. `tf.data.Dataset`은 데이터 프레임, CSV 형식 파일 등과 같은 여러 데이터 소스를 사용합니다.


```python
def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
    def input_function():
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
        if shuffle:
            ds = ds.shuffle(1000)
        ds = ds.batch(batch_size).repeat(num_epochs)
        return ds
    return input_function

train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)
```

다음과 같이 데이터셋을 확인할 수 있습니다:


```python
ds = make_input_fn(dftrain, y_train, batch_size=10)()
for feature_batch, label_batch in ds.take(1):
    print('특성 키:', list(feature_batch.keys()))
    print()
    print('클래스 배치:', feature_batch['class'].numpy())
    print()
    print('레이블 배치:', label_batch.numpy())
```

또한 `tf.keras.layers.DenseFeatures` 층을 사용하여 특정한 피쳐 열의 결과를 확인할 수 있습니다:


```python
age_column = feature_columns[7]
tf.keras.layers.DenseFeatures([age_column])(feature_batch).numpy()
```

`DenseFeatures`는 조밀한(dense) 텐서만 허용합니다. 범주형 데이터를 확인하려면 우선 범주형 열에 `indicator_column` 함수를 적용해야 합니다:


```python
gender_column = feature_columns[0]
tf.keras.layers.DenseFeatures([tf.feature_column.indicator_column(gender_column)])(feature_batch).numpy()
```

모든 기본 피쳐를 모델에 추가한 다음에 모델을 훈련시켜 봅시다. 모델을 훈련하려면 `tf.estimator` API를 이용한 메서드 호출 한번이면 충분합니다:


```python
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)
linear_est.train(train_input_fn)
result = linear_est.evaluate(eval_input_fn)

clear_output()
print(result)
```

### 3.2 도출된 피쳐 열

이제 정확도 76%에 도달했습니다. 별도로 각 기본 피쳐 열을 사용하면 데이터를 설명하기에는 충분치 않을 수 있습니다. 예를 들면, 성별과 레이블간의 상관관계는 성별에 따라 다를 수 있습니다. 따라서 `gender="Male"`과 `gender="Female"`의 단일 모델가중치만 배우면 모든 나이-성별 조합(이를테면 `gender="Male" 그리고 'age="30"` 그리고 `gender="Male"` 그리고 `age="40"`을 구별하는 것)을 포함시킬 수 없습니다.

이처럼 서로 다른 피쳐 조합들 간의 차이를 학습하기 위해서 모델에 **교차 피쳐 열**을 추가할 수 있습니다(또한 교차 열 이전에 나이 열을 버킷화할 수 있습니다):


```python
age_x_gender = tf.feature_column.crossed_column(['age', 'sex'], hash_bucket_size=100)
```

조합한 피쳐를 모델에 추가하고 모델을 다시 훈련시킵니다:


```python
derived_feature_columns = [age_x_gender]
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns+derived_feature_columns)
linear_est.train(train_input_fn)
result = linear_est.evaluate(eval_input_fn)

clear_output()
print(result)
```

이제 정확도 77.6%에 도달했습니다. 기본 피쳐만 이용한 학습보다는 조금 더 좋은 성능을 보여줍니다.

이제 훈련 모델을 이용해서 평가셋에서 승객에 대해 예측을 할 수 있습니다. 텐서플로우 모델은 한번에 샘플의 배치 또는 일부에 대한 예측을 하도록 최적화되어있습니다. 앞서, `eval_input_fn`은 모든 평가셋을 사용하도록 정의되어 있었습니다.


```python
pred_dicts = list(linear_est.predict(eval_input_fn))
probs = pd.Series([pred['probabilities'][1] for pred in pred_dicts])

# 예측 확률 시각화
probs.plot(kind='hist', bins=20, title='predicted probabilities')
```

마지막으로, 수신자 조작 특성(receiver operating characteristic, ROC)을 살펴보면 정탐률(true positive rate)과 오탐률(false positive rate)의 상충관계에 대해 더 잘 이해할 수 있습니다. 


```python
from sklearn.metrics import roc_curve
from matplotlib import pyplot as plt

fpr, tpr, _ = roc_curve(y_eval, probs)
plt.plot(fpr, tpr)
plt.title('ROC curve')
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.xlim(0,)
plt.ylim(0,)
```

# Copyright 2019 The TensorFlow Authors.


```python
#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```
