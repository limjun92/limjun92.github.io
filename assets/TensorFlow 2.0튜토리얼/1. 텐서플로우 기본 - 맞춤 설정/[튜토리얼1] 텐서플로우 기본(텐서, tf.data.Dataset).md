
# [튜토리얼1] 텐서플로우 기본(텐서, tf.data.Dataset)

이번 튜토리얼은 텐서플로우를 사용하기 위한 텐서플로우의 기본을 다루는 입문 튜토리얼입니다. 세부적으로 다음 내용을 다룰 것입니다 : 

* 필요한 패키지 임포트
* 텐서(Tensor) 생성 및 사용
* GPU 가속기 사용
* `tf.data.Dataset` 시연

# 목차
1. 텐서플로우 임포트
2. 텐서(Tensor)
    - 2.1 넘파이 호환성
3. GPU 가속
    - 3.1 장치 이름
    - 3.2 명시적 장치 배치
4. 데이터셋
    - 4.1 소스 데이터셋 생성
    - 4.2 변환 적용
    - 4.3 반복

## 1. 텐서플로우 임포트

시작하기 위해서 텐서플로우 모듈을 임포트합니다. 텐서플로우 2.0에서는 **[즉시 실행(eager execution)](https://www.tensorflow.org/guide/eager)** 이 기본적으로 실행됩니다. 

* **즉시 실행(eager excution)** : 텐서플로우 버젼 2.0에서 새롭게 추가된 기능으로 그래프를 생성하지 않고 함수를 바로 실행하는 명령형 프로그래밍 환경을 의미합니다.


```python
import warnings
warnings.filterwarnings(action='ignore')

import tensorflow as tf
```

## 2. 텐서(Tensor)

[텐서(Tensor)](https://www.tensorflow.org/guide/tensor)는 다차원 배열입니다. 넘파이(NumPy) `ndarray` 객체와 비슷하며, `tf.Tensor` 객체는 데이터 타입과 크기를 가지고 있습니다. 또한 `tf.Tensor`는 GPU와 같은 가속기 메모리에 상주할 수 있습니다. 텐서플로우는 텐서를 생성하고 이용할 수 있는 풍부한 연산 라이브러리(ex. [tf.add](https://www.tensorflow.org/api_docs/python/tf/add), [tf.matmul](https://www.tensorflow.org/api_docs/python/tf/matmul), [tf.linalg.inv](https://www.tensorflow.org/api_docs/python/tf/linalg/inv) 등.)를 제공합니다. 이러한 연산은 자동으로 텐서를 파이썬 네이티브(native) 타입으로 변환합니다.

예를 들어:


```python
# 더하기 연산
print(tf.add(1, 2))
print(tf.add([1, 2], [3, 4]))

# 제곱 계산 연산
print(tf.square(5))

# 총합을 구하는 연산
print(tf.reduce_sum([1, 2, 3]))

# 연산자 오버로딩(overloading) 또한 지원합니다.
print(tf.square(2) + tf.square(3))
```

각각의 `tf.Tensor`는 **크기(shape)** 와 **데이터 타입(dtype)**을 가지고 있습니다.


```python
# 행렬 곱 연산
x = tf.matmul([[1]], [[2, 3]])
print("x : ",x)
print("shape : ",x.shape)
print("dtype : ",x.dtype)
```

넘파이(NumPy) 배열과 `tf.Tensor`의 가장 확연한 차이는 다음과 같습니다:

1. `텐서`는 GPU, TPU와 같은 가속기 메모리에서 사용할 수 있습니다.
2. `텐서`는 불변성(immutable)을 가집니다. 즉, 생성 이후 변경이 불가능한 객체입니다.

### 2.1 넘파이 호환성

텐서와 넘파이 배열 사이의 변환은 다소 간단합니다.

* 텐서플로우 연산은 자동으로 넘파이 배열을 텐서로 변환합니다.
* 넘파이 연산은 자동으로 텐서를 넘파이 배열로 변환합니다.

텐서는 `.numpy()` 메서드(method)를 호출하여 넘파이 배열로 변환할 수 있습니다.
일반적으로는 `tf.Tensor`와 배열은 메모리 표현을 공유하기 때문에 변환이 간단합니다. 그러나 `tf.Tensor`는 GPU 메모리에 저장될 수도 있지만 넘파이 배열은 **항상** 호스트 메모리에 저장되므로, 모든 경우에 변환이 가능한 것은 아닙니다. 따라서 GPU에 저장된 텐서를 넘파이 배열로 변환하기 위해서는 텐서를 호스트 메모리로 복사해야합니다.


```python
import numpy as np

ndarray = np.ones([3, 3])

print("텐서플로우 연산은 자동적으로 넘파이 배열을 텐서로 변환합니다.")
tensor = tf.multiply(ndarray, 42)
print(tensor)

print("\n그리고 넘파이 연산은 자동적으로 텐서를 넘파이 배열로 변환합니다.")
print(np.add(tensor, 1))

print("\n.numpy() 메서드는 텐서를 넘파이 배열로 변환합니다.")
print(tensor.numpy())
```

## 3. GPU 가속

대부분의 텐서플로우 연산은 GPU를 사용하여 가속화됩니다. 따로 어떠한 코드를 명시하지 않아도, 텐서플로우는 연산을 위해 CPU 또는 GPU를 사용할 것인지를 자동으로 결정합니다. 또한, 필요시 텐서를 CPU와 GPU 메모리 사이에서 복사합니다. 연산에 의해 생성된 텐서는 전형적으로 연산이 실행된 장치의 메모리에 의해 실행됩니다. 


```python
x = tf.random.uniform([3, 3])

print("GPU 사용이 가능한가 : "),
print(tf.test.is_gpu_available())
```

-  현재 elice 플랫폼에서는 GPU 사용이 지원되지 않기 때문에 사용이 불가하다는 메시지가 나타납니다.

### 3.1 장치 이름

`Tensor.device`는 텐서를 구성하고 있는 호스트 장치의 풀네임을 제공합니다. 이러한 이름은 프로그램이 실행중인 호스트의 네트워크 주소 및 해당 호스트 내의 장치와 같은 많은 세부 정보를 인코딩하며, 이것은 텐서플로우 프로그램의 분산 실행(이후에 배우게 될 `tf.distribute.Strategy`)에 필요합니다. 예를 들어, 텐서가 호스트의 `N`번째 GPU에 놓여지면 문자열은 `GPU:<N>`으로 끝납니다.


```python
print("텐서가 GPU #0에 있는가 : "),
print(x.device.endswith('GPU:0'))
```

### 3.2 명시적 장치 배치

텐서플로우에서 **배치(replacement)**는 개별 연산을 실행하기 위해 장치에 할당(배치)하는 것입니다. 앞서 언급했듯이, 따로 명시적 지침이 없을 경우 텐서플로우는 연산을 실행하기 위한 장치를 자동으로 결정하고, 필요시 텐서를 장치에 복사합니다. 그러나 텐서플로우 연산은 `tf.device`을 사용하여 특정한 장치에 명시적으로 배치할 수 있습니다. 

- 현재 elice 플랫폼에서 GPU는 지원하지 않지만, GPU가 존재할 시 할당하는 코드는 다음과 같습니다.


```python
import time

def time_matmul(x):
    start = time.time()
    for loop in range(10):
        tf.matmul(x, x)

    result = time.time()-start

    print("10 loops: {:0.2f}ms".format(1000*result))

# CPU에서 강제 실행합니다.
print("On CPU:")
with tf.device("CPU:0"):
    x = tf.random.uniform([1000, 1000])
    # assert 뒤의 조건이 True가 아니라면 AssertError 발생
    assert x.device.endswith("CPU:0")
    time_matmul(x)

# 현재 elice 플랫폼에서는 사용이 불가하지만, 
# 만약 GPU #0 이용 가능시 GPU #0에서 강제 실행하게 명시합니다.
if tf.test.is_gpu_available():
    print("On GPU:")
    with tf.device("GPU:0"): # Or GPU:1 for the 2nd GPU, GPU:2 for the 3rd etc.
        x = tf.random.uniform([1000, 1000])
    # assert 뒤의 조건이 True가 아니라면 AssertError 발생
    assert x.device.endswith("GPU:0")
    time_matmul(x)
```

## 4. 데이터셋

이번에는 모델에 데이터를 제공하기 위한 파이프라인을 구축하기 위해 [`tf.data.Dataset` API](https://www.tensorflow.org/guide/datasets)를 사용해 볼 것입니다.

`tf.data.Dataset` API는 모델을 훈련시키고 평가 루프를 제공할, 간단하고 재사용 가능한 모듈로부터 복잡한 입력 파이프라인을 구축하기 위해 사용됩니다.

### 4.1 소스 데이터셋 생성

소스 데이터셋을 생성할 수 있는 방법에는 다음 2가지가 존재합니다. 

1. 유용한 함수 중 하나인 [`Dataset.from_tensors`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#from_tensors), [`Dataset.from_tensor_slices`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#from_tensor_slices)와 같은 팩토리(factory) 함수 사용

2. 파일로부터 읽어들이는 객체인 [`TextLineDataset`](https://www.tensorflow.org/api_docs/python/tf/data/TextLineDataset) 또는 [`TFRecordDataset`](https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset)를 사용


```python
ds_tensors = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6])

# 출력을 통해 확인하기
print(ds_tensors)
for x in ds_tensors:
    print(x)

print()
    
# CSV 파일을 생성합니다.
import tempfile
_, filename = tempfile.mkstemp()

with open(filename, 'w') as f:
    f.write("""Line 1\nLine 2\nLine 3""")

ds_file = tf.data.TextLineDataset(filename)

# 출력을 통해 확인하기
print(ds_file)
for x in ds_file:
    print(x)
```

### 4.2 변환 적용

[`맵(map)`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map), [`배치(batch)`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#batch), [`셔플(shuffle)`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#shuffle)과 같은 변환 함수를 사용하여 데이터셋의 레코드에 적용합니다. 


```python
ds_tensors = ds_tensors.map(tf.square).shuffle(2).batch(2)

# 출력을 통해 확인하기
print(ds_tensors)
for x in ds_tensors:
    print(x)

print()
    
ds_file = ds_file.batch(2)

# 출력을 통해 확인하기
print(ds_file)
for x in ds_file:
    print(x)
```

### 4.3 반복
`tf.data.Dataset`은 레코드 순회를 지원하는 반복가능한 객체입니다.


```python
print('ds_tensors 요소:')
for x in ds_tensors:
    print(x)

print('\nds_file 요소:')
for x in ds_file:
    print(x)
```

# Copyright 2018 The TensorFlow Authors.


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
