
# [튜토리얼3] 에스티메이터(Estimator)를 활용한 그래디언트 부스팅 모델 만들기

이번 튜토리얼에서는 타이타닉 데이터셋을 가지고 `tf.estimator` API로 의사결정 트리를 사용하여 그래디언트 부스팅(Gradient Boosting) 모델을 훈련시키는 엔드투엔드(end-to-end) 과정을 알아보겠습니다. 부스티드 트리 모델은 회귀와 분류 모두에서 가장 인기 있고 효과적인 머신러닝 접근방식입니다. 이 기법은 여러 트리 모델의 예측을 결합한 앙상블 기법입니다.

부스티드 트리 모델은 최소한의 하이퍼 파라미터(hyperparameter) 튜닝으로 눈에 띄는 성능을 달성할 수 있기 때문에 많은 머신러닝 전문가들이 사용합니다.


```python
from __future__ import absolute_import, division, print_function, unicode_literals

import warnings
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd

from IPython.display import clear_output
from matplotlib import pyplot as plt
import tensorflow as tf
```

# 목차

1. 타이타닉 데이터셋 불러오기
2. (선택 사항) 데이터 탐색
3. 피쳐 열(feature column)과 입력 함수(input function) 생성
4. 모델을 학습시키고 평가하기

## 1. 타이타닉 데이터셋 불러오기

성별, 나이, 클래스 등의 특성을 고려하여 승객 생존을 예측하는 것이 목표인 타이타닉 데이터 세트를 사용할 것입니다.


```python
# 랜덤 시드를 고정해줍니다.
tf.random.set_seed(123)

# 데이터셋을 불러옵니다.
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')
```

데이터셋은 훈련(training) 데이터셋과 검증(evaluation) 데이터셋로 나누어집니다.

* `dftrain`과 `y_train`은 모델에 학습하는 데 사용하는 데이터인 **훈련 데이터**입니다.
* 모델은 **검증 데이터**인 `dfeval`과 `y_eval`로 테스트 됩니다.

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

## 2. (선택 사항) 데이터 탐색

- 이전 튜토리얼과 동일한 데이터를 사용하기 때문에, 이미 학습한 내용이라면 3번 챕터로 넘어가도 무방합니다.

먼저 일부 데이터를 보고 훈련 데이터셋의 요약 통계를 확인해봅시다.


```python
dftrain.head()
```


```python
dftrain.describe()
```

훈련 및 검증 데이터셋에는 각각 627개와 264개의 데이터가 있습니다.


```python
dftrain.shape[0], dfeval.shape[0]
```

아래의 시각화한 자료를 보면 승객의 대다수는 20~30대인 것을 확인할 수 있습니다.


```python
dftrain.age.hist(bins=20)
plt.show()
```

여성 승객보다 남자 승객이 두 배 정도 많습니다.


```python
dftrain.sex.value_counts().plot(kind='barh')
plt.show()
```

많은 승객들의 티켓 등급은 3번째 등급에 속합니다


```python
dftrain['class'].value_counts().plot(kind='barh')
plt.show()
```

대부분의 승객들은 Southampton에서 승선했습니다.


```python
dftrain['embark_town'].value_counts().plot(kind='barh')
plt.show()
```

여성은 남성보다 생존 가능성이 훨씬 높습니다. 이는 분명히 의미있는 피쳐가 될 것입니다.


```python
pd.concat([dftrain, y_train], axis=1).groupby('sex').survived.mean().plot(kind='barh').set_xlabel('% survive')
plt.show()
```

## 3. 피쳐 열(feature column)과 입력 함수(input function) 생성

그래디언트 부스팅 에스티메이터(Gradient Boosting Estimator)는 숫자형 피쳐와 범주형 피쳐를 모두 다룰 수 있습니다. 피쳐 열은 모든 텐서플로우 에스티메이터(Estimator)와 함께 작동하여 모델링에 사용되는 피쳐를 정의합니다. 또한 원-핫 인코딩(One-hot-encoding), 정규화 및 버킷화(bucketization)와 같은 피쳐 엔지니어링(engineering) 기능을 제공합니다. 이 튜토리얼에서는 `CATEGORICAL_COLUMNS`의 필드가 범주형 열에서 원-핫 인코딩된 열로 변환됩니다([indicator column](https://www.tensorflow.org/api_docs/python/tf/feature_column/indicator_column)).


```python
fc = tf.feature_column
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                       'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

def one_hot_cat_column(feature_name, vocab):
    return tf.feature_column.indicator_column(
        tf.feature_column.categorical_column_with_vocabulary_list(feature_name,
                                                 vocab))
feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
    # 범주형 피쳐들은 원-핫 인코딩을 해야합니다.
    vocabulary = dftrain[feature_name].unique()
    feature_columns.append(one_hot_cat_column(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name,
                                           dtype=tf.float32))
```

 `indicator_column`을 이용해서 피쳐 열이 어떻게 변했는지 확인해볼 수 있습니다.


```python
example = dict(dftrain.head(1))
class_fc = tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list('class', ('First', 'Second', 'Third')))
print('Feature value: "{}"'.format(example['class'].iloc[0]))
print('One-hot encoded: ', tf.keras.layers.DenseFeatures([class_fc])(example).numpy())
```

또한 모든 피쳐 열이 어떻게 변했는지 한번에 볼 수도 있습니다.


```python
tf.keras.layers.DenseFeatures(feature_columns)(example).numpy()
```

다음으로는 입력 함수(input function)를 생성하는 과정입니다.

입력 함수는 훈련과 추론을 위해 데이터를 모델로 읽는 방법을 지정해줍니다. 이번 튜토리얼에서는 [`tf.data`](https://www.tensorflow.org/api_docs/python/tf/data) API의 `from_tensor_slices` 메소드를 사용하여 판다스(Pandas)에서 직접 데이터를 읽습니다.이는 데이터셋이 더 작고 인 메모리(in-memory)인 경우에 적합한 방법입니다. 만약 이보다 더 큰 데이터셋의 경우, [csv](https://www.tensorflow.org/api_docs/python/tf/data/experimental/make_csv_dataset)를 포함하는 다양한 파일 형식을 지원하는 tf.data API로 메모리에 맞지 않는 데이터셋을 처리할 수 있습니다.


```python
# 데이터셋이 작으므로 모든 배치 데이터를 사용합니다.
NUM_EXAMPLES = len(y_train)

def make_input_fn(X, y, n_epochs=None, shuffle=True):
    def input_fn():
        dataset = tf.data.Dataset.from_tensor_slices((dict(X), y))
        if shuffle:
            dataset = dataset.shuffle(NUM_EXAMPLES)
        # n_epochs로 지정한만큼 데이터셋을 반복해서 학습합니다.
        dataset = dataset.repeat(n_epochs)
        # 인 메모리 학습은 배칭(batching)를 사용하지 않습니다.
        dataset = dataset.batch(NUM_EXAMPLES)
        return dataset
    return input_fn

# 훈련 데이터와 검증 데이터의 입력 함수입니다.
train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, shuffle=False, n_epochs=1)
```

## 4. 모델을 학습시키고 평가하기

다음 과정을 진행합니다:

1. 피쳐 및 하이퍼파라미터를 지정하여 모델을 초기화합니다.
2. `train_input_fn`로 훈련 데이터를 모델에 적용하고 `train` 함수를 사용하여 모델을 학습시킵니다.
3. 검증 데이터셋으로 모델 성능을 평가합니다. 이 경우에는 `dfeval` 데이터프레임이 이에 해당합니다. 그리고 모델이 예측한 결과가 `y_eval` 배열의 레이블과 얼마나 일치하는지 확인합니다.

부스티드 트리 모델을 학습시켜보기 전에 먼저 선형 분류기인 로지스틱 회귀(Logistic regression) 모델로 학습해봅시다. 비교를 위해서는 간단한 모델부터 시작하는 것이 좋습니다.


```python
linear_est = tf.estimator.LinearClassifier(feature_columns)

# 학습 모델
linear_est.train(train_input_fn, max_steps=100)

# 검증
result = linear_est.evaluate(eval_input_fn)
clear_output()
print(pd.Series(result))
```

이제 부스트 트리 모델로 학습해 보겠습니다. 

부스티드 트리는 회귀(`BoostedTreesRegressor`)와 분류(`BoostedTreesClassifier`)를 지원합니다. 우리는 생존 여부를 예측하는 것이 목표이므로 분류 모델인 `BoostedTreesClassifier`를 사용할 것입니다.





```python
# 메모리가 데이터를 다루는 데 적합하므로 레이어당 전체 데이터셋을 사용하겠습니다. 
# 배치가 하나라는 것은 전체 데이터셋 전체가 하나의 배치라는 것을 의미합니다.
n_batches = 10
est = tf.estimator.BoostedTreesClassifier(feature_columns,
                                          n_batches_per_layer=n_batches)

# 트리가 특정 개수만큼 만들어지면 학습을 멈춥니다. 
est.train(train_input_fn, max_steps=100)

# 검증
result = est.evaluate(eval_input_fn)
clear_output()
print(pd.Series(result))
```

이제 학습된 모델을 사용하여 검증 데이터셋의 승객에 대한 예측을 진행할 수 있습니다. 텐서플로우 모델은 배치 또는 집합(collection) 데이터를 한 번에 예측할 수 있도록 최적화되어있습니다. 앞서 `eval_input_fn`은 전체 검증 데이터셋을 통해 정해집니다.


```python
pred_dicts = list(est.predict(eval_input_fn))
probs = pd.Series([pred['probabilities'][1] for pred in pred_dicts])

probs.plot(kind='hist', bins=20, title='predicted probabilities')
plt.show()
```

마지막으로 결과의 수신기 조작 특성(ROC)도 살펴볼 수 있습니다. 수신기 조작 특성은 TPR(True positive rate)과 FPR(False positive rate)을 더 잘 파악할 수 있습니다.


```python
from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(y_eval, probs)
plt.plot(fpr, tpr)
plt.title('ROC curve')
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.xlim(0,)
plt.ylim(0,)
plt.show()
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
