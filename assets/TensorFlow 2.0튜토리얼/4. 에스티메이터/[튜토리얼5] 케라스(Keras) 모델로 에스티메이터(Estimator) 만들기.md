
# [튜토리얼5] 케라스(Keras) 모델로 에스티메이터(Estimator) 만들기

이번 튜토리얼에는 케라스(Keras)를 이용해서 모델을 만드는 과정을 함께 해볼 것입니다.

텐서플로우 에스티메이터(Estimator)는 텐서플로우에서 완전히 지원되며, 새로운 모델이나 기존에 있던 `tf.keras` 모델로 생성할 수 있습니다. 


```python
from __future__ import absolute_import, division, print_function, unicode_literals

import warnings
warnings.simplefilter('ignore')
import tensorflow as tf

import numpy as np
import tensorflow_datasets as tfds
```

# 목차
1. 간단한 케라스(Keras) 모델 만들기
2. 입력 함수 만들기
3. tf.keras 모델에서 에스티메이터(Estimator) 만들기

### 1. 간단한 케라스(Keras) 모델 만들기

케라스에서는 레이어를 모으고 모델을 제작합니다. 모델은 보통 레이어의 그래프로 가장 일반적인 유형의 모델은 바로 레이어를 쌓는 것입니다.
`tf.keras.Sequential`을 이용해 모델을 만듭니다.

단순하고 완전히 연결된 네트워크(즉, 다중 레이어 인식자)를 구축하려면 다음을 수행합니다:


```python
model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(4,), name ='dense_input'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```

모델을 컴파일하고 모델 구성을 확인하기 위한 모델 요약을 확인합니다.


```python
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.summary()
```

### 2. 입력 함수 만들기

`Datasets` API를 사용하여 대규모 데이터셋이나 다중 장치(multi-device) 학습으로 확장할 수 있습니다.

에스티메이터(Estimator)는 입력 파이프라인을 만드는 시기와 방법을 제어해야 합니다. 이를 위해 **직접 생성한 입력 함수** 또는 `input_fn`이 필요합니다. `Estimator`는 인수 없이 이 함수를 호출합니다. `input_fn`은 `tf.data.Dataset`을 반환해야 합니다.


```python
def input_fn():
    split = tfds.Split.TRAIN
    dataset = tfds.load('iris', split=split, as_supervised=True)
    dataset = dataset.map(lambda features, labels: ({'dense_input':features}, labels))
    dataset = dataset.batch(32).repeat()
    return dataset
```

`input_fn`을 테스트해봅시다.


```python
for features_batch, labels_batch in input_fn().take(1):
    print(features_batch)
    print(labels_batch)
```

### 3. tf.keras 모델에서 에스티메이터(Estimator) 만들기

 `tf.keras.estimator.model_to_estimator`를 이용해 모델을 `tf.estimator.Estimator`로 변환함으로써 `tf.keras.Model`을 `tf.estimator` API로  학습시킬 수 있습니다.


```python
import tempfile
model_dir = tempfile.mkdtemp()

#model_dir = "/tmp/tfkeras_example/"
keras_estimator = tf.keras.estimator.model_to_estimator(
    keras_model=model, model_dir=model_dir)
```

에스티메이터(Estimator)를 학습시키고 평가합니다.


```python
keras_estimator.train(input_fn=input_fn, steps=10)
eval_result = keras_estimator.evaluate(input_fn=input_fn, steps=10)
print('Eval result: {}'.format(eval_result))
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
