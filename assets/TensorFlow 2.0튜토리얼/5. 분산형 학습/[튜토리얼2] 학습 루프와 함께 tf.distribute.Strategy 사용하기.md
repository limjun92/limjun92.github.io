
# [튜토리얼2] 학습 루프와 함께 tf.distribute.Strategy 사용하기

이번 튜토리얼은 사용자 정의 학습 루프(custom training loops)와 함께 [`tf.distribute.Strategy`](https://www.tensorflow.org/guide/distributed_training)를 사용하는 법을 보여드립니다. 우리는 간단한 CNN 모델을 패션 MNIST 데이터셋에 대해 학습할 것입니다. 패션 MNIST 데이터셋은 60000개의 (28 x 28) 크기의 학습 이미지들과 10000개의 (28 x 28) 크기의 테스트 이미지들을 포함하고 있습니다.

이 예제는 유연성을 높이고, 학습을 더 잘 제어할 수 있도록 사용자 정의 학습 루프를 사용합니다. 또한, 사용자 학습 루프를 사용하면 모델과 학습 루프를 디버깅하기 쉬워집니다.

**참고 : 현재 플랫폼에서는 GPU 기능이 지원되지 않습니다! 분산 훈련을 활용하는 방법에 대해 학습하는 목적으로 튜토리얼을 진행해주세요.**


```python
from __future__ import absolute_import, division, print_function, unicode_literals

import warnings
warnings.simplefilter('ignore')
import tensorflow as tf

import numpy as np
import os
```

# 목차
1. 패션 MNIST 데이터셋 다운로드
2. 변수와 그래프를 분산하는 전략 만들기
3. 입력 파이프라인 설정하기
4. 모델 만들기
5. 손실 함수 정의하기
6. 손실과 정확도를 기록하기 위한 지표 정의하기
7. 학습 루프
8. 최신 체크포인트를 불러와서 테스트하기
9. 데이터셋에 대한 반복작업을 하는 다른 방법들

## 1. 패션 MNIST 데이터셋 다운로드


```python
fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


# 하나의 차원을 배열에 추가 -> 새로운 shape == (28, 28, 1)
# 이렇게 하는 이유는 우리의 모델에서 첫 번째 층이 합성곱 층이고
# 합성곱 층은 4D 입력을 요구하기 때문입니다.
# (batch_size, height, width, channels).
# batch_size 차원은 나중에 추가할 것입니다.

train_images = train_images[..., None]
test_images = test_images[..., None]

# 이미지를 [0, 1] 범위로 변경하기.
train_images = train_images / np.float32(255)
test_images = test_images / np.float32(255)
```

## 2. 변수와 그래프를 분산하는 전략 만들기

`tf.distribute.MirroredStrategy` 전략이 어떻게 동작하는지 봅시다:
* 모든 변수와 모델 그래프는 장치(replicas, 다른 문서에서는 replica가 분산 학습에서 장치 등에 복제된 모델을 의미하는 경우가 있으나 이 문서에서는 장치 자체를 의미합니다)에 복제됩니다.
* 입력은 장치에 고르게 분배되어 들어갑니다.
* 각 장치는 주어지는 입력에 대해서 손실(loss)과 그래디언트를 계산합니다.
* 그래디언트들을 전부 더함으로써 모든 장치들 간에 그래디언트들이 동기화됩니다.
* 동기화된 후에, 동일한 업데이트가 각 장치에 있는 변수의 복사본(copies)에 동일하게 적용됩니다.

하나의 범위를 지정해서 모든 코드를 집어넣을 수 있습니다. 아래 코드를 보겠습니다.


```python
# 만약 장치들의 목록이 `tf.distribute.MirroredStrategy` 생성자 안에 명시되어 있지 않다면,
# 자동으로 장치를 인식할 것입니다.
strategy = tf.distribute.MirroredStrategy()
```

* 현재 플랫폼에서 GPU를 사용할 수 없기 때문에 위와 같은 WARNING이 발생하게 됩니다.


```python
print ('장치의 수: {}'.format(strategy.num_replicas_in_sync))
```

## 3. 입력 파이프라인 설정하기

그래프와 변수를 플랫폼과 무관한 SavedModel 형식으로 내보냅니다. 모델을 내보냈다면, 모델을 불러올 때 범위(scope)를 지정하지 않아도 됩니다.


```python
BUFFER_SIZE = len(train_images)

BATCH_SIZE_PER_REPLICA = 64
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

EPOCHS = 10
```

분산 데이터셋들을 `strategy.scope` 내에 생성합니다.


```python
with strategy.scope():

    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE) 
    train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
  
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(GLOBAL_BATCH_SIZE) 
    test_dist_dataset = strategy.experimental_distribute_dataset(test_dataset)
```

## 4. 모델 만들기

`tf.keras.Sequential`을 사용해서 모델을 생성합니다. Model Subclassing API로도 모델 생성을 할 수 있습니다.


```python
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    return model
```


```python
# 체크포인트들을 저장하기 위해서 체크포인트 디렉토리를 생성합니다.
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
```

## 5. 손실 함수 정의하기

일반적으로, GPU/CPU 비율이 1인 단일 장치에서 손실은 입력 배치(batch)의 샘플 개수로 나누어집니다.

그렇다면, `tf.distribute.Strategy`를 사용할 때, 손실은 어떻게 계산되어야 할까요?

* 예를들면, 4개의 GPU가 있고 입력 배치 크기가 64라고 하죠. 입력 배치 하나가 여러 개의 장치(4개의 GPU)에 분배됩니다. 각 장치는 배치 크기가 16인 입력을 받습니다.

* 각 장치에 있는 모델은 해당 입력에 대해 정방향 계산(forward pass)을 수행하고 손실을 계산합니다. 손실을 장치에 할당된 입력 샘플의 수(BATCH_SIZE_PER_REPLICA = 16)로 나누는 것이 아니라 GLOBAL_BATCH_SIZE(64)로 나누어야 합니다.

왜 이렇게 할까요?

* 위와 같이 계산하는 이유는 그래디언트들이 각 장치에서 계산된 다음, 모든 장치를 동기화하기 위해 이 그래디언트 값들을 전부 **더하기** 때문입니다.

이 것을 텐서플로우에서는 어떻게 할까요?


* 만약 이 **예제처럼** 사용자 정의 학습 루프를 **작성한다면**, **다음과 같이 샘플당** 손실을 더하고 GLOBAL_BATCH_SIZE로 **나누어야** 합니다.
`scale_loss = tf.reduce_sum(loss) * (1. / GLOBAL_BATCH_SIZE)`
또는 `tf.nn.compute_average_loss` **함수를** 사용할 수도 있습니다. 이 함수는 **샘플당 손실을 매개변수 값으로 받고 선택적으로 샘플 가중치, GLOBAL_BATCH_SIZE를 받아** 스케일이 조정된 손실을 반환합니다.

* 만약 규제 손실을 사용하는 모델이라면, 장치 개수로 손실 값을 스케일 조정해야 합니다. 이는 `tf.nn_scale_regularization_loss` 함수를 사용하여 처리할 수 있습니다.

* `tf.reduce_mean`을 사용하는 것은 추천하지 않습니다. 이렇게 하면 손실을 실제 장치당 배치 크기로 나눕니다. 이 실제 장치당 배치 크기는 아마 각 단계(step)마다 크기가 다를 수 있습니다.

* 이런 축소(`reduction`)와 스케일 조정은 케라스의 `model.compile`과 `model.fit`에서 자동적으로 수행됩니다.

만약 `tf.keras.losses` 클래스(아래의 예제에서처럼)를 사용한다면, reduction 매개변수를 명시적으로 `NONE` 또는 `SUM` 중 하나로 표시해야 합니다. `AUTO`가 허용되지 않는 이유는 사용자가 분산 모드에서 어떻게 축소할지 명시적으로 지정하는 것이 바람직하기 때문입니다.
현재 `SUM_OVER_BATCH_SIZE`가 장치당 배치 크기로만 나누고 장치 개수로 나누는 것은 사용자에게 위임하기 때문입니다. 그래서 이렇게 하는 것 대신에 사용자가 명시적으로 축소를 수행하도록 만드는 것이 좋습니다.


```python
with strategy.scope():
    # reduction을 `none`으로 설정합니다. 그래서 우리는 축소를 나중에 하고,
    # GLOBAL_BATCH_SIZE로 나눌 수 있습니다.
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        reduction=tf.keras.losses.Reduction.NONE)
    # 또는 loss_fn = tf.keras.losses.sparse_categorical_crossentropy를 사용해도 됩니다.
    def compute_loss(labels, predictions):
        per_example_loss = loss_object(labels, predictions)
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)
```

## 6. 손실과 정확도를 기록하기 위한 지표 정의하기

이 지표(metrics)는 테스트 손실과 학습 정확도, 테스트 정확도를 기록합니다. `.result()`를 사용해서 누적된 통계값들을 언제나 볼 수 있습니다.


```python
with strategy.scope():
    test_loss = tf.keras.metrics.Mean(name='test_loss')

    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_accuracy')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='test_accuracy')
```

## 7. 학습 루프


```python
# 모델과 옵티마이저는 `strategy.scope`에서 만들어져야 합니다.
with strategy.scope():
    model = create_model()

    optimizer = tf.keras.optimizers.Adam()

    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
```


```python
with strategy.scope():
    def train_step(inputs):
        images, labels = inputs

        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = compute_loss(labels, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_accuracy.update_state(labels, predictions)
        return loss 

    def test_step(inputs):
        images, labels = inputs

        predictions = model(images, training=False)
        t_loss = loss_object(labels, predictions)

        test_loss.update_state(t_loss)
        test_accuracy.update_state(labels, predictions)
```


```python
with strategy.scope():
  # `experimental_run_v2`는 주어진 계산을 복사하고,
  # 분산된 입력으로 계산을 수행합니다.
  
    @tf.function
    def distributed_train_step(dataset_inputs):
        per_replica_losses = strategy.experimental_run_v2(train_step,
                                                      args=(dataset_inputs,))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                           axis=None)
 
    @tf.function
    def distributed_test_step(dataset_inputs):
        return strategy.experimental_run_v2(test_step, args=(dataset_inputs,))

    for epoch in range(EPOCHS):
        # 학습 루프
        total_loss = 0.0
        num_batches = 0
        for x in train_dist_dataset:
            total_loss += distributed_train_step(x)
            num_batches += 1
        train_loss = total_loss / num_batches

    # 테스트 루프
    for x in test_dist_dataset:
        distributed_test_step(x)

    if epoch % 2 == 0:
        checkpoint.save(checkpoint_prefix)

    template = ("에포크 {}, 손실: {}, 정확도: {}, 테스트 손실: {}, "
                "테스트 정확도: {}")
    print (template.format(epoch+1, train_loss,
                           train_accuracy.result()*100, test_loss.result(),
                           test_accuracy.result()*100))

    test_loss.reset_states()
    train_accuracy.reset_states()
    test_accuracy.reset_states()
```

위의 예제에서 주목해야 하는 부분은 아래와 같습니다.
* 이 예제는 `train_dist_dataset`과 `test_dist_dataset`을 `for x in ...` 구조를 통해서 반복합니다.
* 스케일이 조정된 손실은 `distributed_train_step`의 반환값입니다. `tf.distribute.Strategy.reduce` 호출해서 장치들 간의 스케일이 조정된 손실 값을 전부 합칩니다. 그리고 나서 `tf.distribute.Strategy.reduce` 반환 값을 더하는 식으로 배치 간의 손실을 모읍니다.
* `tf.keras.Metrics`는 `tf.distribute.Strategy.experimental_run_v2`에 의해서 실행되는 `train_step`과 `test_step` 함수 안에서 업데이트 되어야 합니다.
* `tf.distribute.Strategy.experimental_run_v2`는 그 전략안에 포함된 각 지역 복제 모델로부터 결과값을 반환해 줍니다. 그리고 이 결과를 사용하는 몇 가지 방법들이 있습니다. `tf.distribute.Strategy.reduce`를 이용하여 값들을 합칠 수 있습니다.  `tf.distribute.Strategy.experimental_local_results`를 사용해서 결과값(지역 복제 모델 당 하나의 결과값)에 들어있는 값들의 리스트를 얻을 수도 있습니다.

## 8. 최신 체크포인트를 불러와서 테스트하기

`tf.distribute.Strategy`를 사용해서 체크포인트가 만들어진 모델은 전략 사용 여부에 상관없이 불러올 수 있습니다.


```python
eval_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
      name='eval_accuracy')

new_model = create_model()
new_optimizer = tf.keras.optimizers.Adam()

test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(GLOBAL_BATCH_SIZE)
```


```python
@tf.function
def eval_step(images, labels):
    predictions = new_model(images, training=False)
    eval_accuracy(labels, predictions)
```


```python
checkpoint = tf.train.Checkpoint(optimizer=new_optimizer, model=new_model)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

for images, labels in test_dataset:
    eval_step(images, labels)

print ('전략을 사용하지 않고, 저장된 모델을 복원한 후의 정확도: {}'.format(
    eval_accuracy.result()*100))
```

## 9. 데이터셋에 대해 반복작업을 하는 다른 방법들


### 9.1 반복자(iterator)를 사용하기

주어진 스텝의 수에 따라서 반복하며 전체 데이터셋을 보는 것을 원치 않는다면, `iter`를 호출하여 반복자를 만들 수 있습니다. 그 다음 명시적으로 `next`를 호출합니다. 또한, `tf.funtion` 내부 또는 외부에서 데이터셋을 반복하도록 설정 할 수 있습니다. 다음은 반복자를 사용하여 `tf.function` 외부에서 데이터셋을 반복하는 코드 예제입니다.


```python
with strategy.scope():
    for _ in range(EPOCHS):
        total_loss = 0.0
        num_batches = 0
        train_iter = iter(train_dist_dataset)

    for _ in range(10):
        total_loss += distributed_train_step(next(train_iter))
        num_batches += 1
    average_train_loss = total_loss / num_batches

    template = ("에포크 {}, 손실: {}, 정확도: {}")
    print (template.format(epoch+1, average_train_loss, train_accuracy.result()*100))
    train_accuracy.reset_states()
```

### 9.2 tf.function 내부에서 반복하기
전체 입력 `train_dist_dataset`에 대해서 `tf.function` 내부에서 `for x in ...` 생성자를 사용함으로써 반복을 하거나, 위에서 사용했던 것처럼 반복자를 사용함으로써 반복을 할 수 있습니다. 아래의 예제에서는 `tf.function`로 한 학습의 에포크를 감싸고 그 함수에서 `train_dist_dataset`를 반복하는 것을 보여 줍니다.


```python
with strategy.scope():
    @tf.function
    def distributed_train_epoch(dataset):
        total_loss = 0.0
        num_batches = 0
        for x in dataset:
            per_replica_losses = strategy.experimental_run_v2(train_step,
                                                        args=(x,))
            total_loss += strategy.reduce(
            tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
            num_batches += 1
        return total_loss / tf.cast(num_batches, dtype=tf.float32)

    for epoch in range(EPOCHS):
        train_loss = distributed_train_epoch(train_dist_dataset)

        template = ("에포크 {}, 손실: {}, 정확도: {}")
        print (template.format(epoch+1, train_loss, train_accuracy.result()*100))

    train_accuracy.reset_states()
```

### 9.3 장치 간의 학습 손실 기록하기

참고: 일반적으로, `tf.keras.Metrics`를 사용하여 샘플당 손실 값을 기록하고 장치 내부에서 값이 합쳐지는 것을 피해야 합니다.

`tf.metrics.Mean`을 사용하여 여러 장치의 학습 손실을 기록하는 것을 추천하지 않습니다. 왜냐하면 손실의 스케일을 조정하는 계산이 수행되기 때문입니다.

예를 들어, 다음과 같은 조건의 학습을 수행한다고 합시다.
* 두개의 장치
* 두개의 샘플들이 각 장치에 의해 처리됩니다.
* 손실 값을 산출합니다: 각각의 장치에 대해 [2, 3]과 [4, 5]
* Global batch size = 4

손실의 스케일 조정을 하면, 손실 값을 더하고 GLOBAL_BATCH_SIZE로 나누어 각 장치에 대한 샘플당 손실값을 계산할 수 있습니다. 이 경우에는 (2 + 3) / 4 = 1.24와 (4 + 5) / 4 = 2.25입니다.

하지만 `tf.metrics.Mean`을 사용해서 두 개의 장치에 대해 손실값을 계산한다면, 결과값이 다릅니다. 이 예제에서는, 측정 지표의 `result()`가 메서드가 호출될 때 `total`이 3.50이고 `count`가 2입니다. 결과값은 `total/count`가 1.75가 됩니다. `tf.keras.Metrics`를 이용해서 계산한 손실값이 추가적인 요인에 의해서 크기조정되며, 이 추가적인 요인은 동기화되는 장치의 개수입니다.

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
