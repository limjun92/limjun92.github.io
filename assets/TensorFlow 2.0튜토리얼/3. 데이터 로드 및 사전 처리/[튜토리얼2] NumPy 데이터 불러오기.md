
# [튜토리얼2] NumPy 데이터 불러오기

이번 튜토리얼에서는 NumPy 배열을 `tf.data.Dataset`을 사용하여 로드하는 방법을 다룹니다.

사용하는 데이터셋은 손글씨 데이터인 MNIST 데이터셋으로, ".npz" 형식의 파일에서 로드하여 사용합니다.


```python
import warnings
warnings.simplefilter('ignore')

import numpy as np
import tensorflow as tf
```

# 목차
1. .npz 파일 불러오기 
2. tf. data.Dataset으로 NumPy 배열 불러오기
3. 데이터셋 이용하기
    - 3.1. 데이터셋 셔플(Shuffle) 및 배치(Batch)
    - 3.2 모델 생성과 학습

## 1. `.npz` 파일 불러오기


```python
DATA_URL = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'

path = tf.keras.utils.get_file('mnist.npz', DATA_URL)
with np.load(path) as data:
    train_examples = data['x_train']
    train_labels = data['y_train']
    test_examples = data['x_test']
    test_labels = data['y_test']
```

## 2. `tf.data.Dataset`으로 NumPy 배열 불러오기

예제 데이터 배열과 그에 상응하는 레이블(Label) 배열이 있다면, 두 배열을 `tf.data.Dataset.from_tensor_slices`에 튜플(tuple)로 전달하여 `tf.data.Dataset`으로 생성할 수 있습니다.


```python
train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))
```

## 3. 데이터셋 이용하기

### 3.1 데이터셋 셔플(Shuffle) 및 배치(Batch)


```python
BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)
```

### 3.2 모델 생성과 학습


```python
# tf.keras.Sequential을 통해 층을 쌓아 모델을 생성합니다
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.RMSprop(),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
```


```python
# fit 함수를 통해 모델을 학습시킵니다.
model.fit(train_dataset, epochs=10)
```


```python
model.evaluate(test_dataset)
```

NumPy 배열을 로드하여 95% 정확도를 도출한 것을 확인할 수 있습니다.

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
