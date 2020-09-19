
# [튜토리얼3] pandas.DataFrame 불러오기

이번 튜토리얼에서는 pandas dataframe들을 어떻게 `tf.data.Dataset`로 불러오고 학습시키는 지에 대해 살펴보겠습니다.

이 튜토리얼은 [Cleveland 심장병 클리닉 재단](https://archive.ics.uci.edu/ml/datasets/heart+Disease)이 제공하는 작은 데이터셋를 사용합니다. 

사용하는 데이터 안에는 수백 개의 행이 있는데, 각 행은 **환자에 대한 정보**를 나타내고, 각 열은 그에 대한 **속성**을 나타냅니다. 

우리는 이 정보를 사용하여 환자가 심장병에 걸렸는지 여부를 예측할 것입니다.

이때 이 문제는 병에 걸렸는지, 걸리지 않았는지를 예측하기 때문에 **이진 분류**에 해당합니다.


```python
import warnings
warnings.simplefilter('ignore')

import pandas as pd
import tensorflow as tf
```

# 목차
1. 판다스(pandas)를 이용하여 데이터 읽기
2. tf.data.Dataset을 이용하여 데이터 불러오기
3. 모델을 생성하고 학습시키기
4. 피쳐(Feature) 열에 대한 대안 방법

## 1. 판다스(pandas)를 사용하여 데이터 읽기

이번 튜토리얼에서 사용할 심장에 대한 데이터가 포함된 csv 파일을 다운로드합니다.


```python
csv_file = tf.keras.utils.get_file('heart.csv', 'https://storage.googleapis.com/applied-dl/heart.csv')
```

판다스(pandas)를 이용해서 다운로드 한 csv 파일을 읽습니다.


```python
df = pd.read_csv(csv_file)
```

읽어온 데이터를 확인합니다.


```python
df.head()
```


```python
df.dtypes
```

불러온 데이터 프레임(dataframe) 내의 유일한 `object` 형식인 'thal' 열을 별도의 숫자 값으로 변환합니다.


```python
df['thal'] = pd.Categorical(df['thal'])
df['thal'] = df.thal.cat.codes
```


```python
df.head()
```

## 2. `tf.data.Dataset`을 이용하여 데이터 불러오기

불러온 데이터로 생성한 pandas 데이터 프레임의 값을 읽기 위해 `tf.data.Dataset.from_tensor_slices`를 사용합니다.

`tf.data.Dataset`의 장점 중 하나는 간단하고 매우 효율적인 데이터 파이프라인을 작성할 수 있게 해준다는 것입니다.


```python
target = df.pop('target')
```


```python
dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values))
```


```python
for feat, targ in dataset.take(5):
    print ('Features: {}, Target: {}'.format(feat, targ))
```

`pd.Series`는 `__array__` 프로토콜을 구현하기 때문에 거의 모든 함수에서 `np.array`나 `tf.Tensor`를 이용할 수 있습니다.


```python
tf.constant(df['thal'])
```

데이터셋을 섞고 배치합니다.


```python
train_dataset = dataset.shuffle(len(df)).batch(1)
```

## 3. 모델을 생성하고 학습시키기


```python
def get_compiled_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])
    return model
```


```python
model = get_compiled_model()
model.fit(train_dataset, epochs=15)
```

## 4. 피쳐(Feature) 열에 대한 대안 방법

모델 입력으로 딕셔너리(dictionary)를 전달하는 것은 `tf.keras.layers.Input` 레이어에 대한 딕셔너리(dictionary)를 만드는 것만큼이나 쉽습니다. functional api를 이용하면 그 어떤 전처리 과정도 적용할 수 있고 이를 레이어로 쌓을 수도 있습니다. 

이는 피쳐 열 대신 사용할 수 있을 것입니다.


```python
inputs = {key: tf.keras.layers.Input(shape=(), name=key) for key in df.keys()}
x = tf.stack(list(inputs.values()), axis=-1)

x = tf.keras.layers.Dense(10, activation='relu')(x)
output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model_func = tf.keras.Model(inputs=inputs, outputs=output)

model_func.compile(optimizer='adam',
                   loss='binary_crossentropy',
                   metrics=['accuracy'])
```

`tf.data`를 사용할 때 `pd.DataFrame`의 열의 형태를 보존하는 가장 쉬운 방법은 `pd.DataFrame`을 `dictionary` 형식으로 변환하고, 변환한 딕셔너리(dictionary)을 슬라이싱(Slice)하여 사용하는 것입니다.


```python
dict_slices = tf.data.Dataset.from_tensor_slices((df.to_dict('list'), target.values)).batch(16)
```


```python
for dict_slice in dict_slices.take(1):
    print (dict_slice)
```


```python
model_func.fit(dict_slices, epochs=15)
```

# Copyright 2019 The TensorFlow Authors.

Licensed under the Apache License, Version 2.0 (the "License");


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
