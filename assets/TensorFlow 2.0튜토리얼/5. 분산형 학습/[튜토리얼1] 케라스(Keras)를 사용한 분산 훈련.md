
# [튜토리얼1] 케라스(Keras)를 사용한 분산 훈련

이번 튜토리얼에서는 MNIST 데이터셋을 가지고 `tf.distribute.MirroredStrategy`를 사용하여 모델을 학습시키는 방법에 대해서 알아보도록 하겠습니다.

`tf.distribute.Strategy` API는 훈련을 여러 처리 장치들로 분산시키는 것을 추상화한 것입니다. 기존의 모델이나 훈련 코드를 조금만 바꾸어 분산 훈련을 할 수 있게 하는 것이 분산 전략 API의 목표입니다.

이 튜토리얼에서는 `tf.distribute.MirroredStrategy`를 사용합니다. 이 전략은 동기화된 훈련 방식을 활용하여 한 장비에 있는 여러 개의 GPU로 그래프 내 복제를 수행합니다. 즉, 모델의 모든 변수를 각 프로세서에 복사합니다. 그리고 각 프로세서의 그래디언트(gradient)를 [올 리듀스(all-reduce)](http://mpitutorial.com/tutorials/mpi-reduce-and-allreduce/)를 사용하여 모읍니다. 그다음 모아서 계산한 값을 각 프로세서의 모델 복사본에 적용합니다.

`MirroredStategy`는 텐서플로에서 기본으로 제공하는 몇 가지 분산 전략 중 하나입니다.

**참고 : 현재 플랫폼에서는 GPU 기능이 지원되지 않습니다. 분산 훈련을 활용하는 방법에 대해 학습하는 목적으로 튜토리얼을 진행해주세요.**


```python
from __future__ import absolute_import, division, print_function, unicode_literals

import warnings
warnings.simplefilter('ignore')

import tensorflow_datasets as tfds
import tensorflow as tf
tfds.disable_progress_bar()

import os
```

# 목차
1. 데이터셋 다운로드
2. 분산 전략 정의하기
3. 입력 파이프라인 구성하기
4. 모델 만들기
5. 콜백 정의하기
6. 학습과 평가

## 1. 데이터셋 다운로드

MNIST 데이터셋을 [TensorFlow Datasets](https://www.tensorflow.org/datasets)에서 다운로드받은 후 불러옵니다. 이 함수는 `tf.data` 형식을 반환합니다.

`with_info`를 `True`로 설정하면 전체 데이터에 대한 메타 정보도 함께 불러옵니다. 이 정보는 `info` 변수에 저장됩니다. 여기에는 훈련과 테스트 샘플 수를 비롯한 여러가지 정보들이 들어있습니다.


```python
datasets, info = tfds.load(name='mnist', with_info=True, as_supervised=True)

mnist_train, mnist_test = datasets['train'], datasets['test']
```

## 2. 분산 전략 정의하기

분산과 관련된 처리를 하는 `MirroredStrategy` 객체를 만듭니다. 이 객체가 컨텍스트 관리자(`tf.distribute.MirroredStrategy.scope`)도 제공하는데, 이 안에서 모델을 만들어야 합니다.


```python
strategy = tf.distribute.MirroredStrategy()
```

* 현재 플랫폼에서 GPU를 사용할 수 없기 때문에 위와 같은 WARNING이 발생하게 됩니다.


```python
print('장치의 수: {}'.format(strategy.num_replicas_in_sync))
```

## 3. 입력 파이프라인 구성하기

다중 GPU로 모델을 훈련할 때는 배치 크기를 늘려야 컴퓨팅 자원을 효과적으로 사용할 수 있습니다. 기본적으로는 GPU 메모리에 맞추어 가능한 가장 큰 배치 크기를 사용하십시오. 이에 맞게 학습률도 조정해야 합니다.


```python
# 데이터셋 내 샘플의 수는 info.splits.total_num_examples 로도
# 얻을 수 있습니다.

num_train_examples = info.splits['train'].num_examples
num_test_examples = info.splits['test'].num_examples

BUFFER_SIZE = 10000

BATCH_SIZE_PER_REPLICA = 64
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
```

픽셀의 값은 0~255 사이이므로 0-1 범위로 정규화해야 합니다. 정규화 함수를 정의합니다.


```python
def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255

    return image, label
```

이 함수를 훈련과 테스트 데이터에 적용합니다. 훈련 데이터 순서를 섞고, [훈련을 위해 배치로 묶습니다](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#batch).


```python
train_dataset = mnist_train.map(scale).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
eval_dataset = mnist_test.map(scale).batch(BATCH_SIZE)
```

## 4. 모델 만들기

`strategy.scope` 컨텍스트 안에서 케라스 모델을 만들고 컴파일합니다.


```python
with strategy.scope():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(loss='sparse_categorical_crossentropy',
                optimizer=tf.keras.optimizers.Adam(),
                metrics=['accuracy'])
```

## 5. 콜백 정의하기

여기서 사용하는 콜백은 다음과 같습니다.

*   **텐서보드(TensorBoard)**: 이 콜백은 텐서보드용 로그를 남겨서, 텐서보드에서 그래프를 그릴 수 있게 해줍니다.
*   **모델 체크포인트(Checkpoint)**: 이 콜백은 매 에포크(epoch)가 끝난 후 모델을 저장합니다.
*   **학습률 스케줄러**: 이 콜백을 사용하면 매 에포크 혹은 배치가 끝난 후 학습률을 바꿀 수 있습니다.

콜백을 추가하는 방법을 보여드리기 위하여 **학습률**을 표시하는 콜백도 추가하겠습니다.


```python
# 체크포인트를 저장할 체크포인트 디렉터리를 지정합니다.
checkpoint_dir = './training_checkpoints'
# 체크포인트 파일의 이름
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
```


```python
# 학습률을 점점 줄이기 위한 함수
# 필요한 함수를 직접 정의하여 사용할 수 있습니다.
def decay(epoch):
    if epoch < 3:
        return 1e-3
    elif epoch >= 3 and epoch < 7:
        return 1e-4
    else:
        return 1e-5
```


```python
# 에포크가 끝날 때마다 학습률을 출력하는 콜백.
class PrintLR(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print('\n에포크 {}의 학습률은 {}입니다.'.format(epoch + 1,
                                                      model.optimizer.lr.numpy()))
```


```python
callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
    tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                                       save_weights_only=True),
    tf.keras.callbacks.LearningRateScheduler(decay),
    PrintLR()
]
```

## 6. 학습과 평가

이제 평소처럼 모델을 학습합시다. 모델의 `fit` 함수를 호출하고 튜토리얼의 시작 부분에서 만든 데이터셋을 넘깁니다. 이 단계는 분산 훈련 여부와 상관없이 동일합니다.


```python
model.fit(train_dataset, epochs=12, callbacks=callbacks)
```

아래에서 볼 수 있듯이 체크포인트가 저장되고 있습니다.


```python
# 체크포인트 디렉터리 확인하기
!ls {checkpoint_dir}
```

모델의 성능이 어떤지 확인하기 위하여, 가장 최근 체크포인트를 불러온 후 테스트 데이터에 대하여 `evaluate`를 호출합니다.

평소와 마찬가지로 적절한 데이터셋과 함께 `evaluate`를 호출하면 됩니다.


```python
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

eval_loss, eval_acc = model.evaluate(eval_dataset)

print('평가 손실: {}, 평가 정확도: {}'.format(eval_loss, eval_acc))
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
