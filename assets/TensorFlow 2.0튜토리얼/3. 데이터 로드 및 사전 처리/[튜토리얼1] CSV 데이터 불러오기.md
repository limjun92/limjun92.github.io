
# [튜토리얼1] CSV 데이터 불러오기

이번 튜토리얼에서는 **CSV 데이터 파일**을 `tf.data.Dataset`으로  불러와 타이타닉의 생존자를 예측해보겠습니다.

이번 튜토리얼에서 사용할 데이터는 타이타닉에 탑승했던 승객들에 대한 정보가 담겨있는 데이터입니다. 승객의 나이, 성별, 티켓의 등급, 혼자 여행하는 승객인지에 대한 여부 등의 특징들을 이용하여 승객의 생존 여부를 예측해보겠습니다.


```python
# 필요한 라이브러리 임포트
import warnings
warnings.simplefilter('ignore')

import numpy as np
import tensorflow as tf
```

# 목차
1. 데이터 불러오기
2. 데이터 전처리
3. 모델 생성하기
    - 3.1. 연속형(Continuous) 데이터
        - 데이터 정규화(Normalization)
    - 3.2. 범주형(Categorical) 데이터
    - 3.3 복합전처리층
4. 모델 생성하기
5. 학습하고 평가하고 예측하기

## 1. 데이터 불러오기

먼저 CSV 파일인 타이타닉 데이터(훈련 데이터, 테스트 데이터)의 파일 경로를 불러오겠습니다.


```python
TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"

train_file_path = tf.keras.utils.get_file("train.csv", TRAIN_DATA_URL)
test_file_path = tf.keras.utils.get_file("eval.csv", TEST_DATA_URL)
```

불러온 CSV 파일의 경로를 이용하여 첫 부분을 보고 어떻게 구성되어있는지 확인합니다.:


```python
!head {train_file_path}
```

판다스(pandas)를 이용해서 CSV 데이터를 불러와 넘파이(NumPy) 배열로 변환하여 텐서플로우에서 이용할 수도 있습니다. 만약 대규모 데이터를 사용하거나 텐서플로우나 tf.data 형식으로 통합해야 할 때에는 `tf.data.experimental.make_csv_dataset` 함수를 사용하면 됩니다.

모델이 예측해야할 값을 가진 열(Column)은 **생존여부(`survived`)** 로 레이블(label) 값은 **0은 사망, 1은 생존**을 의미합니다.


```python
LABEL_COLUMN = 'survived'
LABELS = [0, 1]
```

이제 본격적으로 CSV 데이터를 불러와 데이터셋을 만들어 봅시다.


```python
def get_dataset(file_path, **kwargs):
    dataset = tf.data.experimental.make_csv_dataset(
      file_path,
      batch_size=5, # 예제를 쉽게 만들기 위해서 작은 배치 사이즈를 사용합니다.
      label_name=LABEL_COLUMN,
      na_value="?",
      num_epochs=1,
      ignore_errors=True, 
      **kwargs)
    return dataset

raw_train_data = get_dataset(train_file_path)
raw_test_data = get_dataset(test_file_path)
```


```python
def show_batch(dataset):
    for batch, label in dataset.take(1):
        for key, value in batch.items():
            print("{:20s}: {}".format(key,value.numpy()))
```

데이터셋에 있는 각각의 항목(Item)은 (**많은 예시들**, **많은 레이블들**)로 이루어진 튜플을 나타내는 배치(Batch)입니다. **예시들** 안의 데이터는 배치 크기 만큼의 요소를 가진 열 기반의 텐서들이 구성되어 있습니다.

이를 자세히 보면 아래와 같습니다.


```python
show_batch(raw_train_data)
```

CSV의 열들은 모두 이름이 있습니다. 데이터셋을 만드는 사람은 이 열 이름들을 자동으로 정할 수 있습니다. 만약 파일의 첫 번째 열의 이름이 없는 경우, 열의 이름을 순서대로 나열한 리스트(list)를 `make_csv_dataset` 함수 내에 있는 인자 `column_names`의 값으로 전달해서 지정할 수 있습니다.


```python
CSV_COLUMNS = ['survived', 'sex', 'age', 'n_siblings_spouses', 'parch', 'fare', 'class', 'deck', 'embark_town', 'alone']

temp_dataset = get_dataset(train_file_path, column_names=CSV_COLUMNS)

show_batch(temp_dataset)
```

이번 예제에서는 가능한 데이터 내의 모든 열들을 사용할 것입니다. 그러나 만약 데이터셋에서 특정 열을 제외하고 싶다면, 사용하고 싶은 열들의 이름만 리스트(list)로 만들어서 `select_columns`의 값으로 전달하면 됩니다. 


```python
SELECT_COLUMNS = ['survived', 'age', 'n_siblings_spouses', 'class', 'deck', 'alone']

temp_dataset = get_dataset(train_file_path, select_columns=SELECT_COLUMNS)

show_batch(temp_dataset)
```

## 3. 데이터 전처리

CSV 파일은 다양한 타입의 데이터를 다룰 수 있습니다. 일반적인 경우 모델에 데이터를 학습시키기 전에 여러 데이터 타입이 혼합된 데이터를 길이가 고정된 벡터로 변환합니다.

텐서플로우에서는 `tf.feature_column` 함수가 내장되어 있어 위와 같은 데이터를 변환시킬 수 있습니다. 또한, nltk나 sklearn와 같은 다양한 도구로  데이터들을 전처리하고, 전처리를 완료한 데이터를 텐서플로우에서 이용할 수도 있습니다.

만약 `tf.feature_column` 함수를 사용하여 텐서플로우 모델 내에서 전처리 하는 경우 다음과 같은 다양한 장점이 존재합니다. 

**모델 내에서 전처리하는 것의 장점**
1. 모델을 외부로 전달하거나 이동시킬 때에도 **모델 내에 전처리 과정이 포함되어 있습니다**.

2. 모델에 전처리 하지 않은 **raw 상태의 데이터도 그대로 적용**할 수도 있습니다.

### 3.1 연속형(Continuous) 데이터

데이터가 이미 적절한 숫자형식(numeric)으로 되어있다,면 이를 그대로 벡터로 전달하여 모델에 적용할 수 있습니다.


```python
SELECT_COLUMNS = ['survived', 'age', 'n_siblings_spouses', 'parch', 'fare']
DEFAULTS = [0, 0.0, 0.0, 0.0, 0.0]
temp_dataset = get_dataset(train_file_path, 
                           select_columns=SELECT_COLUMNS,
                           column_defaults = DEFAULTS)

show_batch(temp_dataset)
```


```python
example_batch, labels_batch = next(iter(temp_dataset)) 
```

다음은 모든 열들을 하나로 합치는 기능을 구현한 간단한 함수입니다.


```python
def pack(features, label):
    return tf.stack(list(features.values()), axis=-1), label
```

구현한 함수를 데이터셋에 적용합니다.


```python
packed_dataset = temp_dataset.map(pack)

for features, labels in packed_dataset.take(1):
    print(features.numpy())
    print()
    print(labels.numpy())
```

만약 데이터가 다양한 데이터 타입으로 이루어져 있다면, `tf.feature_column` api를 통해서 숫자형 데이터만 따로 다룰 수 있습니다.그러나 이 방법은 **오버헤드**를 발생시킬 수 있기 때문에 정말 필요한 상황이 아니라면 사용하지 않는 것을 추천합니다.


```python
show_batch(raw_train_data)
```


```python
example_batch, labels_batch = next(iter(temp_dataset)) 
```

위에서 소개한 방법보다 좀 더 일반화된 전처리로 숫자형 열들을 골라내고 골라낸 열들을 하나의 열로 합치는 방법은 다음과 같습니다.


```python
class PackNumericFeatures(object):
    def __init__(self, names):
        self.names = names

    def __call__(self, features, labels):
        numeric_features = [features.pop(name) for name in self.names]
        numeric_features = [tf.cast(feat, tf.float32) for feat in numeric_features]
        numeric_features = tf.stack(numeric_features, axis=-1)
        features['numeric'] = numeric_features

        return features, labels
```


```python
NUMERIC_FEATURES = ['age','n_siblings_spouses','parch', 'fare']

packed_train_data = raw_train_data.map(
    PackNumericFeatures(NUMERIC_FEATURES))

packed_test_data = raw_test_data.map(
    PackNumericFeatures(NUMERIC_FEATURES))
```


```python
show_batch(packed_train_data)
```


```python
example_batch, labels_batch = next(iter(packed_train_data)) 
```

#### - 데이터 정규화(Normalization)

연속형 데이터는 대부분 정규화(Normalization)를 적용하는 것이 좋습니다.


```python
import pandas as pd
desc = pd.read_csv(train_file_path)[NUMERIC_FEATURES].describe()
desc
```


```python
MEAN = np.array(desc.T['mean'])
STD = np.array(desc.T['std'])
```


```python
def normalize_numeric_data(data, mean, std):
  # Center the data
  return (data-mean)/std
```

정규화를 적용하여 숫자형 열을 만들어봅시다. `tf.feature_columns.numeric_column` API에는 각 배치마다 정규화를 적용할 수 있도록 하는 인자, `normalizer_fn`가 있습니다.

[`functools.partial`](https://docs.python.org/3/library/functools.html#functools.partial)을 이용해서 정규화 함수에 `MEAN`과 `STD`를 설정해줍니다.


```python
import functools

normalizer = functools.partial(normalize_numeric_data, mean=MEAN, std=STD)

numeric_column = tf.feature_column.numeric_column('numeric', normalizer_fn=normalizer, shape=[len(NUMERIC_FEATURES)])
numeric_columns = [numeric_column]
numeric_column
```

When you train the model, include this feature column to select and center this block of numeric data:
모델을 훈련시킬 때, 이 피쳐(Feature) 열을 포함하여 이 숫자형 데이터 블록을 선택하고 가운데 놓습니다.


```python
example_batch['numeric']
```


```python
numeric_layer = tf.keras.layers.DenseFeatures(numeric_columns)
numeric_layer(example_batch).numpy()
```

이와 같은 평균 기반 정규화를 위해서는 각 열의 평균을 미리 알아야 합니다.

### 3.2 범주형(Categorical) 데이터

CSV 데이터 중 일부 열들은 범주형(Categorical) 데이터입니다.

각 범주형 데이터 열마다 `tf.feature_column.indicator_column`을 가진 컬렉션을 만들기 위해서는 `tf.feature_column` API를 사용해야 합니다.




```python
CATEGORIES = {
    'sex': ['male', 'female'],
    'class' : ['First', 'Second', 'Third'],
    'deck' : ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
    'embark_town' : ['Cherbourg', 'Southhampton', 'Queenstown'],
    'alone' : ['y', 'n']
}
```


```python
categorical_columns = []
for feature, vocab in CATEGORIES.items():
    cat_col = tf.feature_column.categorical_column_with_vocabulary_list(
        key=feature, vocabulary_list=vocab)
    categorical_columns.append(tf.feature_column.indicator_column(cat_col))
```


```python
# 만든 열을 확인해봅시다.
categorical_columns
```


```python
categorical_layer = tf.keras.layers.DenseFeatures(categorical_columns)
print(categorical_layer(example_batch).numpy()[0])
```

이는 나중에 모델을 만들 때 데이터 전처리 입력의 일부가 될 것입니다. 

### 3.3 복합전처리층

두 가지 입력 유형(연속형, 범주형 데이터 타입)을 모두 추출하고 전처리하기 위한 입력 레이어를 만들기 위해서는 `tf.keras.layers.DenseFeatures`에 두 열을 합쳐 전달하면 됩니다.


```python
preprocessing_layer = tf.keras.layers.DenseFeatures(categorical_columns+numeric_columns)
```


```python
print(preprocessing_layer(example_batch).numpy()[0])
```

## 4. 모델 생성하기


생성한 `preprocessing_layer`을 첫 층으로 하여 쌓는 `tf.keras.Sequential` 모델을 생성합니다.


```python
model = tf.keras.Sequential([
  preprocessing_layer,
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])
```

## 5. 학습하고 평가하고 예측하기

이제 모델을 객체화하고 학습시킬 수 있습니다.


```python
train_data = packed_train_data.shuffle(500)
test_data = packed_test_data
```


```python
model.fit(train_data, epochs=20)
```

모델이 학습을 완료하면 **테스트 데이터**에 대한 정확도를 확인할 수 있습니다.


```python
test_loss, test_accuracy = model.evaluate(test_data)

print('\n\nTest Loss {}, Test Accuracy {}'.format(test_loss, test_accuracy))
```

테스트 데이터의 배치 데이터 세트에 대한 레이블들을 예측하기 위해서 `tf.keras.Model.predict`를 사용합니다.


```python
predictions = model.predict(test_data)

# 일부 결과를 확인해보겠습니다.
for prediction, survived in zip(predictions[:10], list(test_data)[0][1][:10]):
    print("Predicted survival: {:.2%}".format(prediction[0]),
        " | Actual outcome: ",
        ("SURVIVED" if bool(survived) else "DIED"))


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
