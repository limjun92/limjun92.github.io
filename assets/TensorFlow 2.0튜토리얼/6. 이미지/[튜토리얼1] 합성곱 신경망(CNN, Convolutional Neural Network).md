
# [튜토리얼1] 합성곱 신경망(CNN, Convolutional Neural Network)

이 튜토리얼은 **이미지 데이터**인 MNIST 숫자를 분류하기 위해 간단한 **합성곱 신경망(Convolutional Neural Network, CNN)** 을 훈련합니다. 간단한 이 네트워크는 MNIST 테스트 세트에서 99%의 정확도를 달성할 것입니다. 이 튜토리얼은 [케라스 Sequential API](https://www.tensorflow.org/guide/keras)를 사용하기 때문에 몇 줄의 코드만으로 모델을 만들고 훈련할 수 있습니다.


```python
from __future__ import absolute_import, division, print_function, unicode_literals

import warnings
warnings.simplefilter('ignore')

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
```

# 목차
1. MNIST 데이터셋 다운로드하고 준비하기
2. 합성곱 층 만들기
3. 마지막에 Dense 층 추가하기
4. 모델 컴파일과 훈련하기
5. 모델 평가

### 1. MNIST 데이터셋 다운로드하고 준비하기

(28,28) 크기의 60,000개의 학습 데이터셋과 10,000개의 테스트 데이터셋을 다운받습니다.


```python
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

print("학습 데이터 shape : ", train_images.shape)
print("테스트 데이터 shape :", test_images.shape)
```

불러온 학습 데이터 중 10번째 데이터를 확인해보도록 하겠습니다.


```python
plt.imshow(train_images[10], cmap='gray')
```

모델 학습을 위해 데이터에 채널을 추가하고, 각 픽셀 값을 0과 1 사이로 정규화합니다

각 픽셀 값은 0~255 사이의 숫자값을 가지기 때문에 255로 나누어줄 경우 0~1 사이의 값을 가지게 됩니다. 


```python
# 컬러 채널 추가(흑백이기 때문에 1)
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# 픽셀 값을 0~1 사이로 정규화합니다. 
train_images, test_images = train_images / 255.0, test_images / 255.0
```

### 2. 합성곱 층 만들기

아래 6줄의 코드에서 [Conv2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D)와 [MaxPooling2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D) 층을 쌓는 일반적인 패턴으로 합성곱 층을 정의합니다.

CNN은 배치(batch) 크기를 제외하고 (이미지 높이, 이미지 너비, 컬러 채널) 크기의 텐서(tensor)를 입력으로 받습니다. MNIST 데이터는 (흑백 이미지이기 때문에) 컬러 채널(channel)이 하나지만 컬러 이미지는 (R,G,B) 세 개의 채널을 가집니다. 이 예에서는 MNIST 이미지 포맷인 (28, 28, 1) 크기의 입력을 처리하는 CNN을 정의하겠습니다. 이 값을 첫 번째 층의 `input_shape` 매개변수로 전달합니다.


```python
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))  
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
```

지금까지 모델의 구조를 출력해 봅시다.


```python
model.summary()
```

위에서 Conv2D와 MaxPooling2D 층의 출력은 (높이, 너비, 채널) 크기의 3D 텐서입니다. 높이와 너비 차원은 네트워크가 깊어질수록 감소하는 경향을 가집니다. Conv2D 층에서 출력 채널의 수는 첫 번째 매개변수에 의해 결정됩니다(예를 들면, 32 또는 64). 일반적으로 높이와 너비가 줄어듦에 따라 (계산 비용 측면에서) Conv2D 층의 출력 채널을 늘릴 수 있습니다.

### 3. 마지막에 Dense 층 추가하기

모델을 완성하려면 마지막 합성곱 층의 출력 텐서(크기 (3, 3, 64))를 하나 이상의 Dense 층에 주입하여 분류를 수행합니다. Dense 층은 벡터(1D)를 입력으로 받는데 현재 출력은 3D 텐서입니다. 먼저 3D 출력을 1D로 펼치겠습니다. 그다음 하나 이상의 Dense 층을 그 위에 추가하겠습니다. MNIST 데이터는 10개의 클래스가 있으므로 마지막에 Dense 층에 10개의 출력과 소프트맥스 활성화 함수를 사용합니다.


```python
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
```

최종 모델의 구조를 확인해 봅시다.


```python
model.summary()
```

여기에서 볼 수 있듯이 두 개의 Dense 층을 통과하기 전에 (3, 3, 64) 출력을 (576) 크기의 벡터로 펼쳤습니다.

### 4. 모델 컴파일과 훈련하기


```python
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)
```

### 5. 모델 평가


```python
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
```


```python
print("테스트 정확도 : %d" %(test_acc*100))
```

결과에서 보듯이 간단한 CNN 모델이 99%의 테스트 정확도를 달성합니다.

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
