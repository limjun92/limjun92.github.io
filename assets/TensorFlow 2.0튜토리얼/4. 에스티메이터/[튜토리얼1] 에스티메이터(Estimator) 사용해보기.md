
# [튜토리얼1] 에스티메이터(Estimator) 사용해보기

이번 튜토리얼에서는 텐서플로우의 [**에스티메이터(Estimator)**](https://www.tensorflow.org/guide/estimator) 를 사용하여 붓꽃(Iris) 데이터 분류 문제를 해결하는 방법을 설명합니다. 

* **에스티메이터(Estimator)** 는 텐서플로우의 전체 모델을 개괄적으로 표현한 것으로, 간편한 확장과 비동기적인 훈련을 할 수 있도록 설계되었습니다.


```python
from __future__ import absolute_import, division, print_function, unicode_literals

import warnings
warnings.simplefilter('ignore')

import pandas as pd
import tensorflow as tf
```

# 목차
1. 데이터셋
2. 에스티메이터(Estimator)를 이용한 프로그래밍 개요
3. 입력 함수 생성하기
4. 피쳐 열 정의하기
5. 에스티메이터(Estimator)의 인스턴스화
6. 훈련하고, 평가하고, 예측하기
    - 6.1 모델 훈련시키기
    - 6.2 훈련된 모델 평가하기
    - 6.3 학습된 모델로 예측하기

## 1. 데이터셋

이 예제에서는 붓꽃 데이터를 사용하여 각각의 꽃받침과 꽃잎의 크기에 따라 세 개의 다른 종으로 분류하는 모델을 만들고 테스트합니다.

붓꽃(iris) 데이터셋에는 네 가지 피쳐과 하나의 레이블이 있습니다. 네 가지 피쳐는 개별 붓꽃의 식물학적 특성을 식별하는데 사용하고, 자세한 항목은 다음과 같습니다. 

* 꽃받침(Sepal) 길이
* 꽃받침(Sepal) 너비
* 꽃잎(Petal) 길이
* 꽃잎(Petal) 너비

따라서 데이터 내에 존재하는 피쳐의 이름과 분류해야 하는 붓꽃의 종인 레이블은 다음과 같이 정의할 수 있습니다. 


```python
# 피쳐의 이름(꽃받침 길이, 꽃받침 너비, 꽃잎 길이, 꽃잎 너비)
CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']

# 분류해야 하는 붓꽃의 종()
SPECIES = ['Setosa', 'Versicolor', 'Virginica']
```

케라스(Keras)와 판다스(Pandas)를 사용하여 붓꽃 데이터셋을 다운로드합니다. 

훈련을 위한 데이터셋과 테스트를 위한 데이터셋을 구분해야 합니다.


```python
train_path = tf.keras.utils.get_file(
    "iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = tf.keras.utils.get_file(
    "iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
```

다운받은 데이터를 확인하면 4개의 실수 피쳐와 1개의 int32형의 레이블이 있는 것을 확인할 수 있습니다.


```python
train.head()
```

각 데이터셋에 대해 레이블을 구분하여 예측할 수 있도록 모델을 훈련합니다.


```python
# 레이블 열은 피쳐에서 제거합니다
train_y = train.pop('Species')
test_y = test.pop('Species')

train.head()
```

## 2. 에스티메이터(Estimator)를 이용한 프로그래밍 개요

이제 데이터가 설정되었으므로 텐서플로우 에스티메이터(Estimator)를 사용하여 모델을 정의할 수 있습니다. 에스티메이터(Estimator)는 `tf.estimator.Estimator`에서 파생된 클래스입니다. 텐서플로우는 일반적인 ML 알고리즘을 구현하기 위한 `tf.estimator`(예: `LinearRegressor`)의 집합(collection)을 제공합니다. 그 외에도 사용자가 지정하여 에스티메이터(Estimator)를 만들 수 있습니다. 이제 막 시작하는 분들은 이미 만들어진 에스티메이터(Estimator)를 사용하는 것이 좋습니다.

만들어진 에스티메이터(Estimator)를 가지고 텐서플로우 프로그램을 작성하려면 다음과 같은 작업을 수행해야 합니다:

* 하나 이상의 입력 함수를 생성합니다.
* 모델의 피쳐 열을 정의합니다.
* 피쳐 열을 비롯한 다양한 기능을 지정하여 에스티메이터(Estimator)를 [객체(instance)](https://ko.wikipedia.org/wiki/%EC%9D%B8%EC%8A%A4%ED%84%B4%EC%8A%A4_(%EC%BB%B4%ED%93%A8%ED%84%B0_%EA%B3%BC%ED%95%99))화 합니다.
* 적절한 입력 함수를 데이터 소스로 전달하여 에스티메이터(Estimator) 객체에서 하나 이상의 메서드를 호출합니다.

이러한 작업들을 통해 어떻게 모델이 붓꽃 품종을 분류하는지 살펴보겠습니다.

## 3. 입력 함수 생성하기

입력 함수를 만들어 학습하고, 평가하고, 예측하기 위한 데이터를 적용시켜야 합니다.

**입력 함수**는 `tf.data.Dataset` 객체를 반환하는 함수로 아래와 같은 두 원소를 가진 튜플을 생성합니다:

* `features` - 다음과 같은 파이썬 딕셔너리입니다:
    * 각 key값은 피쳐의 이름입니다.
    * 각 value값은 해당하는 key값의 값들을 가집니다.
* `label` - 모든 샘플들의 레이블 값을 가진 배열입니다.

간단한 방법으로 입력 함수의 형식을 살펴보겠습니다:


```python
def input_evaluation_set():
    # 딕셔너리 생성
    features = {'SepalLength': np.array([6.4, 5.0]),
                'SepalWidth':  np.array([2.8, 2.3]),
                'PetalLength': np.array([5.6, 3.3]),
                'PetalWidth':  np.array([2.2, 1.0])}
    # 레이블 배열 생성
    labels = np.array([2, 1])
    
    return features, labels
```

이제 입력 함수를 통해 원하는 대로 `features` 딕셔너리와 `label` 목록을 생성할 수 있습니다. 

하지만 이 방법보다 모든 종류의 데이터를 분석할 수 있는 텐서플로우의 [Dataset API](https://www.tensorflow.org/guide/datasets)를 사용하는 것이 더 좋습니다.

일반적인 경우, Dataset API는 오류없이 이를 처리할 수 있습니다. 예를 들어, Dataset API를 사용하면 병렬적으로 대용량 파일 모음에서 레코드를 쉽게 읽어들일 수 있고, 이를 단일 스트림에 결합할 수 있습니다.

데이터를 판다스(pandas)로 불러오고 메모리 내 데이터로 입력 파이프라인을 구축해 이 예제를 간단히 만들어봅시다.



```python
def input_fn(features, labels, training=True, batch_size=256):
    """훈련과 평가를 위한 입력 함수"""
    # 입력 데이터를 데이터셋으로 변환합니다.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # 훈련 모드라면 반복해서 섞습니다.
    if training:
        dataset = dataset.shuffle(1000).repeat()
    
    return dataset.batch(batch_size)
```

## 4. 피쳐 열 정의하기

피쳐 열은 모델에서 어떻게 피쳐 딕셔너리의 가공되지 않은(Raw) 입력 데이터를 사용하는 지에 대해 설명하는 객체입니다. 에스티메이터(Estimator) 모델을 만들 때 모델에 사용할 각 피쳐를 설명하는 피쳐 열의 목록을 이 모델에 전달합니다. [`tf.feature_column`](https://www.tensorflow.org/api_docs/python/tf/feature_column) 모듈은 모델에 데이터를 나타내는 다양한 옵션들을 제공합니다.

붓꽃 데이터의 경우, 4개의 가공되지 않은(Raw) 피쳐는 **숫자형(numeric)** 값이기 때문에 에스티메이터(Estimator) 모델이 각 4개의 피쳐들을 32비트의 부동 소수점(floating-point) 값으로 나타내도록 하는 피쳐 열의 목록을 만들 것입니다.

피쳐 열을 생성하는 코드는 다음과 같습니다:


```python
# 피쳐 열은 입력으로 사용하는 방법을 보여줍니다.
my_feature_columns = []
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))
```

이제 우리는 모델이 가공되지 않은(Raw) 피쳐를 나타내도록 할 수 있으므로 에스티메이터(Estimator)를 만들 수 있습니다.

(사실 피쳐 열은 이번 튜토리얼에서 사용되는 것보다 훨씬 정교하게 사용될 수 있으며, 4장 [튜토리얼1] 정형데이터 다루기에서 더 자세하게 다루겠습니다.)

## 5. 에스티메이터(Estimator)의 객체화

붓꽃 데이터 문제는 전형적인 분류 문제입니다. 텐서플로우는 다음과 같은 몇 가지 미리 만들어진 분류기 에스티메이터(Classifier Estimator)를 제공합니다.

* `tf.estimator.DNNClassifier`는 멀티클래스로 이루어진 깊은 모델을 위한 것입니다.
* `tf.estimator.DNNLinearCombinedClassifier`는 넓고 깊은 모델을 위한 것입니다.
* `tf.estimator.LinearClassifier`는 선형 모델을 기반으로 하는 분류기를 위한 것입니다.

붓꽃 데이터 문제에서는 `tf.estimator.DNNClassifier`가 가장 적절해보입니다. 이 에스티메이터(Estimator)를 객체화하는 방법은 아래와 같습니다.


```python
# 각각 30개, 10개의 노드로 구성된 2층의 은닉 층을 쌓은 DNN을 생성합니다.
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    # 두 개의 은닉 층의 노드의 수를 각각 30개와 10개로 설정합니다.
    hidden_units=[30, 10],
    # 모델은 3개의 클래스 중 하나를 선택해야 합니다. 
    n_classes=3)
```

## 6. 훈련하고, 평가하고, 예측하기

이제 에스티메이터(Estimator) 객체가 있으므로 메서드를 호출하여 다음 작업을 수행할 수 있습니다.

* 모델 훈련시키기
* 훈련된 모델을 평가하기
* 훈련된 모델로 예측하기

### 6.1 모델 훈련시키기

다음과 같이 에스티메이터(Estimator)의 `train` 메서드를 호출하여 모델을 훈련시킬 수 있습니다.

* `input_fn` : Estimator가 예상한대로 인수를 사용하지 않는 입력 함수를 제공하면서 인수를 캡처하기 위해 [`lambda`](https://docs.python.org/3/tutorial/controlflow.html)에서 input_fn을 호출합니다.
* `steps` : 몇번의 훈련 후 모델 훈련을 중지할지 설정합니다. 


```python
# 모델 훈련시키기
classifier.train(
    input_fn=lambda: input_fn(train, train_y, training=True),
    steps=5000)
```

### 6.2 훈련된 모델 평가하기

이제 모델이 훈련되었으므로 모델의 성능에 대한 통계를 얻을 수 있습니다. 다음 코드에서는 테스트 데이터에 대한 모델의 정확도를 측정합니다.



```python
eval_result = classifier.evaluate(
    input_fn=lambda: input_fn(test, test_y, training=False))

print('\n테스트 셋 정확도: {accuracy:0.3f}\n'.format(**eval_result))
```

`train` 메서드를 호출하는 것과 달리, 테스트 셋의 정확도를 평가할 때는 `steps` 인수를 전달하지 않았습니다. 평가를 위한 `input_fn`은 오직 하나의 에포크(epoch)의 데이터만을 생성합니다.

`eval_result` 딕셔너리에는 `average_loss`(샘플당 평균 손실)와 `loss`(미니 배치당 평균 손실), 에스티메이터의 `global_step`(훈련 반복 횟수)도 포함되어 있습니다.

### 6.3 학습된 모델로 예측하기

좋은 평가 결과를 생성하는 모델로 훈련되었습니다. 이제 훈련된 모델을 사용하여 레이블이 없는 붓꽃의 종을 예측할 수 있습니다. 훈련시키고 평가했던 것과 마찬가지로, 하나의 함수를 호출하여 예측합니다:


```python
# 모델에서 예측값을 생성합니다.
expected = ['Setosa', 'Versicolor', 'Virginica']
predict_x = {
    'SepalLength': [5.1, 5.9, 6.9],
    'SepalWidth': [3.3, 3.0, 3.1],
    'PetalLength': [1.7, 4.2, 5.4],
    'PetalWidth': [0.5, 1.5, 2.1],
}

def input_fn(features, batch_size=256):
    """예측을 위한 입력 함수"""
    # 입력값을 레이블이 없는 데이터셋으로 변환합니다.
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

predictions = classifier.predict(
    input_fn=lambda: input_fn(predict_x))
```

`predict` 메서드는 Python으로 반복될 수 있고(iterable), 만들 수 있는 각 샘플에 대한 예측 결과를 가진 딕셔너리를 반환합니다.
다음 코드는 일부 예측 결과와 그 확률을 보여줍니다:


```python
for pred_dict, expec in zip(predictions, expected):
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]

    print('예측 결과 : "{}" ({:.1f}%), 실제 결과 : "{}"'.format(
        SPECIES[class_id], 100 * probability, expec))
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
