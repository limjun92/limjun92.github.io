
# [튜토리얼3] 사전학습된 합성곱 신경망(CNN)을 이용한 전이학습


이번 튜토리얼에서는 사전 학습된 신경망 전이 학습을 사용하여 고양이와 개의 이미지를 분류하는 방법을 배우게 됩니다.


사전 학습 된 모델은 이전에 대규모 데이터 세트, 일반적으로 대규모 이미지 분류 작업에서 훈련 되어 저장된 신경망입니다. 사전 학습 된 모델을 그대로 사용하거나 전이 학습을 사용하여 이 모델을 주어진 작업에 맞게 사용자 정의합니다.

이미지 분류를 위한 전이 학습은 모델이 충분히 크고 일반적인(general) 데이터셋에 대해 학습하면 이 모델은 시각적 세계의 일반적인 모델로써 효과적으로 작용한다는 것입니다. 대규모 데이터셋에서 대규모 모델을 처음부터 학습시킬 필요 없이 이러한 학습된 피쳐 맵을 활용할 수 있습니다.

이번 튜토리얼에서는 사전 학습된 모델을 사용자 지정(customize)하는 두 가지 방법을 시도해 봅니다.

1.  **Feature Extraction(피쳐 추출)**<br>
이전 신경망에서 학습된 표현을 사용하여 새 샘플의 의미 있는 피쳐를 추출합니다. 사전 학습된 모델 위에 처음부터 학습할 새 분류기를 추가하기만 하면 이전 데이터셋에 대해 학습한 피쳐 맵을 재사용 할 수 있습니다.<br>
  전체 모델을 (재)학습시킬 필요는 없습니다. 기본 합성곱 신경망에는 이미 사진을 분류하는 데 일반적으로 유용한 기능이 포함되어 있습니다. 그러나 사전 훈련된 모델의 최종 분류 부분은 원래의 분류 작업과 모델이 학습한 클래스 집합에 한정됩니다.

2. **Fine-Tuning(미세 조정)**<br> 
고정 된 모델베이스의 일부 최상위 레이어를 프리즈한 것을 해제(unfreeze)하고 새로 추가 된 분류기 레이어와 기본 모델의 마지막 레이어를 함께 학습합니다 이를 통해 기본 모델에서 상위 피쳐를 "미세 조정"하여 특정 작업에 보다 적합하게 만들 수 있습니다.

다음과 같은 일반적인 머신 러닝 작업 흐름(workflow)을 따릅니다.

1. 데이터 검토 및 이해
2. Keras ImageDataGenerator를 사용한 입력 파이프라인 빌드
3. 모델 구성
   *   사전 학습된 기본(base) 모델과 가중치 로드하기
   *   상위에 분류 레이어 쌓기
4. 모델 학습
5. 모델 평가 




```python
import warnings
warnings.simplefilter('ignore')

import tensorflow as tf

import os

import numpy as np

import matplotlib.pyplot as plt

from tensorflow import keras
```

# 목차
1. 데이터 전처리
    - 1.1 데이터 다운로드하기
    - 1.2 데이터 형태 맞추기
2. 사전 학습된 합성곱 신경망으로부터 베이스 모델 생성하기
3. 피쳐 추출하기
    - 3.1 컨볼루셔널 베이스 프리즈하기
    - 3.2 분류 헤드 추가하기
    - 3.3 모델 컴파일하기
    - 3.4 모델 학습하기
    - 3.5 학습 곡선
4. 미세 조정(Fine tuning)
    - 4.1 모델의 상단 레이어 프리즈 해제
    - 4.2 모델 컴파일하기
    - 4.3 모델 계속 학습시키기
5. 요약

## 1. 데이터 전처리

### 1.1 데이터 다운로드하기

고양이와 강아지 데이터셋을 불러오기 [TensorFlow Datasets](http://tensorflow.org/datasets)을 사용합니다.

이 `tfds(TensorFlow Datasets)` 패키지는 사전 정의 된 데이터를 로드하는 가장 쉬운 방법입니다.





```python
# 텐서플로우 데이터 셋 임포트하기
import tensorflow_datasets as tfds
```

이 `tfds.load` 메소드는 데이터를 다운로드하여 캐시하고 `tf.data.Dataset` 객체를 리턴 합니다. 이러한 객체는 데이터를 처리하고 모델에 연결하는 강력하고 효율적인 방법을 제공합니다.

`"cats_vs_dogs"` 데이터셋은 표준 분할을 정의하지 않으므로 하위 분할 함수인 `subsplit` 메소드을 사용하여 각각 데이터의 80%, 10%, 10% 로 (학습 train, 유효성 검사 validation, 테스트 test)로 나눕니다.


```python
SPLIT_WEIGHTS = (8, 1, 1)
splits = tfds.Split.TRAIN.subsplit(weighted=SPLIT_WEIGHTS)

(raw_train, raw_validation, raw_test), metadata = tfds.load(
    'cats_vs_dogs', split=list(splits),
    with_info=True, as_supervised=True)
```

위 코드 실행의 결과로 'tf.data.Dataset' 객체는 `(이미지, 레이블)' 쌍을 포함합니다. 여기서 이미지는 다양한 모양과 3개의 채널을 가지고 있으며(?, ?, 3), 레이블은 스칼라 값입니다. 




```python
print(raw_train)
print(raw_validation)
print(raw_test)
```

학습 데이터셋의 처음 두 개의 이미지와 레이블을 확인해 보겠습니다. 


```python
get_label_name = metadata.features['label'].int2str

for image, label in raw_train.take(2):
    plt.figure()
    plt.imshow(image)
    plt.title(get_label_name(label))
```

### 1.2 데이터 형태 맞추기

데이터 형태를 맞추는 `format_example` 함수를 작성합니다. 

`tf.image` 모듈을 사용하여 분류 작업을 위한 이미지 형태를 맞춥니다. 

이미지를 고정된 입력 크기로 조정하고 입력 채널을 `[-1,1]` 범위로 재조정합니다.

<!-- TODO(markdaoust): fix the keras_applications preprocessing functions to work in tf2 -->


```python
IMG_SIZE = 100 # 모든 이미지들은 고정된 입력 크기인 100x100 사이즈로 조정합니다.

def format_example(image, label):
    image = tf.cast(image, tf.float32)
    image = (image/127.5) - 1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label
```

이 함수를 `map` 메소드를 사용하여 데이터셋의 각 항목에 적용합니다. 


```python
train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)
```

이제 데이터를 섞고 배치(batch)시킵니다.


```python
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000
```


```python
train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)
```

데이터의 배치를 확인합니다:


```python
for image_batch, label_batch in train_batches.take(1):
    pass

image_batch.shape
```

## 2. 사전 학습 된 합성곱 신경망으로부터 베이스 모델 생성하기

Google에서 개발한 **MobileNet V2** 모델로 기본 모델을 생성합니다. 이 모델은 140만개의 이미지와 1000개의 클래스로 구성된 대규모 데이터셋인 ImageNet 데이터셋으로 사전에 학습된 것입니다.ImageNet은 "열대과일 잭프룻"과 "주사기"와 같은 다양한 범주를 가진 연구 학습 데이터셋입니다.

먼저 피쳐 추출에 사용할 MobileNet V2 레이어를 선택해야합니다. 마지막 분류 레이어(대부분의 기계 학습 모델이 아래에서 위로 이동하므로 "상단"에 있는)은 그다지 유용하지 않습니다. 대신, 평탄화(Flatten) 작업을 수행하기 직전 마지막 레이어에 의존하기 위한 일반적인 관행을 따릅니다. 이 레이어를 "병목(bottleneck) 레이어"라고 합니다. 이러한 병목 레이어는 최종 / 상위 레이어와 비교하여 좀 더 일반성(generality)을 유지한다는 특징이 있습니다.

이제 ImageNet 에서 학습한 가중치가 사전로드된 MobileNet V2 모델을 객체화합니다. **include_top=False** 인자를 지정하면 맨 위에 분류 레이어가 포함되지 않은 모델을 불러오므로 피쳐 추출에 이상적입니다.


```python
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

# 사전 학습된 모델 MobileNet V2로 베이스 모델을 생성합니다.
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
```

이 피쳐 추출기는 각 '100x100x3' 이미지를 '4x4x1280'의 피쳐 블록으로 변환합니다. 예제 이미지 배치에 어떤 영향을 미치는지 출력을 통해 확인해 보겠습니다.



```python
feature_batch = base_model(image_batch)
print(feature_batch.shape)
```

## 3. 피쳐 추출하기

이 단계에서는 이전 단계에서 생성된 컨볼루셔널 베이스를 프리즈(freeze)하고 피쳐 추출기로 사용합니다. 또한 그 위에 분류기를 추가하고 최상위 분류기를 학습시킵니다.

### 3.1 컨볼루셔널 베이스 프리즈하기
모델을 컴파일하고 학습시키기 전에 컨볼루셔널 베이스를 프리즈하는 것이 중요합니다. `base_model.trainable` 를 통해 학습 중에 특정 레이어의 가중치가 업데이트되는 것을 방지합니다. MobileNet V2에는 여러 레이어가 있으므로 전체 모델의 학습 가능한 플래그를 False로 설정하면 모든 레이어가 프리즈됩니다.


```python
base_model.trainable = False
```


```python
# 베이스 모델 구조를 확인합니다.
base_model.summary()
```

### 3.2 분류 헤드 추가하기

피쳐 블록으로부터 예측값을 생성하기 위해, `tf.keras.layers.GlobalAveragePooling2D` 레이어를 사용하여 이미지 내의 `4x4 '공간 위치의 평균을 계산하여 특징을 이미지 당 단일 1280 요소 벡터로 변환합니다.


```python
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)
```

`tf.keras.layers.Dense` 레이어를 적용하여 이러한 피쳐들을 이미지 당 단일 예측값으로 변환합니다. 이 예측값은 로짓(logit)이나 raw 예측 값으로 취급되므로 여기서는 활성화 함수가 필요하지 않습니다. 양수는 클래스 1을 예측하고 음수는 클래스 0을 예측합니다.



```python
prediction_layer = keras.layers.Dense(1)
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)
```

이제 `tf.keras.Sequential` 모델을 사용하여 피쳐 추출기와 이 두 레이어를 쌓습니다.


```python
model = tf.keras.Sequential([
  base_model,
  global_average_layer,
  prediction_layer
])
```

### 3.3 모델 컴파일하기
모델 학습 전 먼저 모델을 컴파일해야 합니다. 이 데이터셋은 개와 고양이라는 두 개의 레이블을 가지고 있으므로 이진 교차 엔트로피 손실(binary cross-entropy loss)을 사용하며, 선형 출력값을 제공하기 때문에 `from_logits= True` 로 설정합니다.


```python
base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
```


```python
model.summary()
```

MobileNet의 250만개의 파라미터는 프리즈되지만, Dense 레이어에는 1.200개의  학습 가능한(Trainable) 파라미터가 존재합니다. 이것들은 두 개의 `tf.Variable` 객체, 즉 가중치와 편향(biases)으로 나누어집니다.



```python
len(model.trainable_variables)
```

### 3.4 모델 학습하기

2 에포크에 대해 학습한 후에는 최대 86%의 정확도를 확인할 수 있습니다.



```python
num_train, num_val, num_test = (
  metadata.splits['train'].num_examples*weight/10
  for weight in SPLIT_WEIGHTS
)
```


```python
initial_epochs = 2
steps_per_epoch = round(num_train)//BATCH_SIZE
validation_steps=20

loss0,accuracy0 = model.evaluate(validation_batches, steps = validation_steps)
```


```python
print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))
```


```python
history = model.fit(train_batches,
                    epochs=initial_epochs,
                    validation_data=validation_batches)
```

### 3.5 학습 곡선
MobileNet V2 기본 모델을 고정 피쳐 추출기로 사용할 때의 학습 및 검증 정확도/손실의 학습 곡선을 살펴보겠습니다.


```python
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()
```

참고: 검증(validation) 메트릭스가 학습 메트릭스보다 확실히 나은 주요 원인은 `tf.keras.layers.BatchNormalization`와 `tf.keras.layers.Dropout`가 학습 중 정확도에 영향을 미치기 때문입니다. 검증(validation) 손실을 계산할 때 이 기능이 사용되지 않습니다.

또한 학습 메트릭스는 각 에포크(epoch)의 평균을 보고하는 반면 검증 메트릭스에서는 에포크 후 평가되므로 검증 메트릭스는 학습 메트릭스보다 약간 더 오래 학습된 모델을 보기 때문입니다.

## 4. 미세 조정(Fine tuning)

위 피쳐 추출 부분에서는 MobileNet V2 기본 모델을 기반으로 몇 개의 계층만 학습시켰기 때문에 사전 학습된 신경망의 가중치는 학습 중에 업데이트되지 않았습니다.

성능을 더욱 향상시키는 한 가지 방법은 추가한 분류기의 학습과 함께 사전 학습된 모델 최상위 레이어의 가중치를 학습(혹은 미세 조정)시키는 것입니다. 학습 과정에서는 일반 피쳐 맵에서 데이터셋과 관련된 피쳐로 가중치가 강제 조정됩니다.

참고: 사전 학습된 모델이 학습 불가능으로 설정한 최상위 분류기를 학습시킨 후에만 이 작업을 수행해야 합니다. 사전 학습된 모델 위에 랜덤하게 초기화된 분류기를 추가하고 모든 레이어를 공동으로 학습하려고 하면 그래디언트 업데이트의 크기가 너무 커지며(분류기에서 무작위 가중치로 인해) 사전 학습된 모델은 학습한 내용을 잊어버립니다.

또한 전체 MobileNet 모델보다 소수의 상위 레이어를 미세 조정(fine-tuning)해야 합니다. 대부분의 컨볼루셔널 신경망에서는 레이어가 높을수록 전문화됩니다. 처음 몇 개 레이어는 거의 모든 유형의 이미지에 일반화하는 매우 간단하고 일반적인 피쳐를 배웁니다. 더 높이 올라갈수록 모델이 학습된 데이터셋에 대한 피쳐가 점점 더 구체화됩니다. 미세 조정의 목표는 일반적인 학습 내용을 덮어쓰기보다는 이러한 전문화된 피쳐를 새로운 데이터셋과 함께 작동하도록 조정하는 것입니다.

### 4.1 모델의 상단 레이어 프리즈 해제

`base_model`의 프리즈를 해제하고 하단 레이어를 학습할 수 없도록 설정하면 됩니다. 그런 다음 모델을 다시 컴파일하고(이러한 변경 사항을 적용하기 위해 필요) 학습을 재개해야 합니다.


```python
base_model.trainable = True
```


```python
# 베이스 모델에 몇개의 레이어가 있는지 확인해봅시다.
print("Number of layers in the base model: ", len(base_model.layers))

# 이후 미세 조정 할 기준 레이어를 설정합니다.
fine_tune_at = 100

# 설정한 기준 레이어 전의 모든 레이어는 프리즈합니다.
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable =  False
```

### 4.2 모델 컴파일하기
훨씬 낮은 학습률을 사용하여 모델을 컴파일합니다.


```python
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer = tf.keras.optimizers.RMSprop(lr=base_learning_rate/10),
              metrics=['accuracy'])
```


```python
model.summary()
```


```python
len(model.trainable_variables)
```

### 4.3 모델 계속 학습시키기

앞서 수렴(convergence)를 학습 시키고 난 후 이 단계를 수행하면 정확도가 몇 퍼센트 향상됩니다.


```python
fine_tune_epochs = 2
total_epochs =  initial_epochs + fine_tune_epochs

history_fine = model.fit(train_batches,
                         epochs=total_epochs,
                         initial_epoch =  history.epoch[-1],
                         validation_data=validation_batches)
```

MobileNet V2 기본 모델의 마지막 몇 레이어를 미세 조정하고 그 위에 분류기를 학습시킬 때 학습 및 검증 정확도/손실의 학습 곡선을 살펴보겠습니다. 검증 손실은 학습 손실보다 훨씬 더 높기 때문에 다소 오버 피팅이 발생할 수 있습니다.

또한 새로운 학습 데이터셋이 상대적으로 작고 원래의 MobileNet V2 데이터셋과 유사하기 때문에 다소 오버 피팅이 발생할 수도 있습니다.


미세 조정 후 모델은 거의 90%의 정확도에 도달합니다.


```python
acc += history_fine.history['accuracy']
val_acc += history_fine.history['val_accuracy']

loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']
```


```python
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0.8, 1])
plt.plot([initial_epochs-1,initial_epochs-1],
          plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 1.0])
plt.plot([initial_epochs-1,initial_epochs-1],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()
```

## 5. 요약

* **피쳐 추출에 사전 학습된 모델 사용**: 소규모 데이터셋으로 작업할 때는 동일한 도메인의 대규모 데이터셋에서 학습한 모델을 활용하는 것이 일반적인 방법입니다. 이 작업은 사전 학습된 모델을 인스턴스화하고 위에 완전히 연결된 분류기를 추가하여 수행됩니다. 사전 학습된 모델은 "프리즈"되며, 훈련 중에 분류기의 가중치만 업데이트됩니다.
이번 예시에서 컨볼루셔널 베이스 모델은 각 이미지와 관련된 모든 피쳐를 추출하고 해당 추출된 피쳐 집합에 따라 이미지 클래스를 결정하는 분류기를 학습했습니다.

* **사전 학습된 모델 조정**: 성능을 더욱 개선하기 위해 사전 학습된 모델의 최상위 레이어의 용도를 미세 조정을 통해 새 데이터셋으로 변경해야 할 수 있습니다.
우리는 모델에서 데이터셋과 관련된 고급 피쳐를 학습하도록 가중치를 조정했습니다. 이 기술은 일반적으로 학습 데이터셋이 크고 사전 학습된 모델이 학습한 원래 데이터셋과 매우 유사한 경우에 권장됩니다.

##### Copyright 2019 The TensorFlow Authors.


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


```python
#@title MIT License
#
# Copyright (c) 2017 François Chollet                                                                                                                    # IGNORE_COPYRIGHT: cleared by OSS licensing
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
```
