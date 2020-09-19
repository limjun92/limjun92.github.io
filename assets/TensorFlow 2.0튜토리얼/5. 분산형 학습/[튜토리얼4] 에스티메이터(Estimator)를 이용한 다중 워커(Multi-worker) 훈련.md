
# [튜토리얼4] 에스티메이터(Estimator)를 이용한 다중 워커(Multi-worker) 훈련

이번 튜토리얼에서는 `tf.estimator`를 이용한 분산형 다중 워커 학습을 하기 위해 `tf.distribute.Strategy`를 사용하는 방법을 보여줍니다. `tf.estimator`를 사용하여 코드를 작성하고 단일 장비(machine)보다 뛰어난 성능을 내려고 할 경우 이 튜토리얼을 따라하면 좋습니다.

참고: `tf.distribute` API로 에스티메이터(Estimator)를 사용할 수 있지만, 대신 `tf.distribute`가 있는 케라스(Keras)를 사용하는 것이 좋습니다. 이 때 `tf.distribute.Strategy`를 사용한 에스티메이터 학습은 제한적으로 지원되고 있습니다.

**참고 : 현재 플랫폼에서는 GPU 기능이 지원되지 않습니다! 분산 훈련을 활용하는 방법에 대해 학습하는 목적으로 튜토리얼을 진행해주세요.**


```python
from __future__ import absolute_import, division, print_function, unicode_literals

import warnings
warnings.simplefilter('ignore')

import tensorflow_datasets as tfds
import tensorflow as tf
tfds.disable_progress_bar()

import os, json
```

# 목차
1. 입력 함수
2. 다중 워커 설정
3. 모델 정의
4. MultiWorkerMirroredStrategy
5. 모델 학습과 평가
6. 학습 성능 최적화

## 1. 입력 함수

이번 튜토리얼에서는 텐서플로우 데이터셋에 있는 MNIST 데이터셋을 사용합니다. 

다중 워커 학습에 에스티메이터를 사용할 때 모델을 통합하려면 워커의 수를 기준으로 데이터셋을 샤딩(sharding)해야 합니다. 

입력 데이터는 워커 인덱스에 의해 샤딩되기 때문에 각 워커가 데이터셋의 `1/num_workers`를 구분하여 처리하도록 합니다.


```python
BUFFER_SIZE = 10000
BATCH_SIZE = 64

def input_fn(mode, input_context=None):
    datasets, info = tfds.load(name='mnist',
                                with_info=True,
                                as_supervised=True)
    mnist_dataset = (datasets['train'] if mode == tf.estimator.ModeKeys.TRAIN else
                   datasets['test'])

    def scale(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255
        return image, label

    if input_context:
        mnist_dataset = mnist_dataset.shard(input_context.num_input_pipelines,
                                        input_context.input_pipeline_id)
    return mnist_dataset.map(scale).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
```

통합하기 위한 또 다른 합리적인 방법은 각 워커의 고유한 시드(sead)로 데이터셋을 섞는 것입니다.

## 2. 다중 워커 설정

`TF_CONFIG` 환경 변수는 클러스터의 일부인 각 워커 별로 클러스터의 설정(configuration)을 지정하는 표준 방법입니다.

`TF_CONFIG`에는 `cluster`와 `task`라는 두 가지 요소가 있습니다. 

* `cluster`는 전체 클러스터, 즉 클러스터의 작업자 및 매개 변수 서버에 대한 정보를 제공합니다. 
* `task`는 현재 태스크에 대한 정보를 제공합니다. 

이 예에서`type`은 `worker`, `index`는 `0`입니다.

이 튜토리얼에서는 `localhost`의 `TF_CONFIG`에 2개의 워커를 설정하는 방법을 볼 것입니다. 실제로 외부 IP 주소와 포트에 여러개의 워커를 생성하고 각 워커의 `TF_CONFIG`를 올바르게 설정합니다. 즉, `index` 작업을 수정합니다.

```
os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ["localhost:12345", "localhost:23456"]
    },
    'task': {'type': 'worker', 'index': 0}
})
```



## 3. 모델 정의

학습을 위한 레이어, 옵티마이저(optimizer) 및 손실 함수를 작성합니다.


```python
LEARNING_RATE = 1e-4
def model_fn(features, labels, mode):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    logits = model(features, training=False)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'logits': logits}
        return tf.estimator.EstimatorSpec(labels=labels, predictions=predictions)

    optimizer = tf.compat.v1.train.GradientDescentOptimizer(
          learning_rate=LEARNING_RATE)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(
          from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(labels, logits)
    loss = tf.reduce_sum(loss) * (1. / BATCH_SIZE)
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=optimizer.minimize(
          loss, tf.compat.v1.train.get_or_create_global_step()))
```

참고: 이 예에서는 학습 속도가 고정되어 있지만 일반적으로는 글로벌 배치(batch) 크기에 따라 학습 속도를 조정해야 합니다.

## 4. MultiWorkerMirroredStrategy

모델을 학습시키려면 `tf.distribute.experimental.MultiWorkerMirroredStrategy` 객체를 사용합니다. `MultiWorkerMirroredStrategy`는 모든 워커의 각 장치에 있는 모델의 레이어 내의 모든 변수의 복사본을 생성합니다. 또한 집단 통신(collective communication)을 위한 텐서플로우 op인 `CollectiveOps`를 사용하여 그래디언트(gradient)를 집계하고 변수를 동기화 상태로 유지합니다.


```python
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
```

## 5. 모델 학습과 평가

다음으로, 에스티메이터의 `RunConfig`에 분산 전략(distribution strategy)을 지정하고, `tf.estimator.train_and_evaluate`를 호출하여 학습하고 평가합니다. 이번 튜토리얼에서는 `train_distribute`를 통해 전략을 지정하여 학습만 분산합니다. 또한 `eval_distribute`를 통해 평가를 분산할 수도 있습니다.


```python
config = tf.estimator.RunConfig(train_distribute=strategy)

classifier = tf.estimator.Estimator(
    model_fn=model_fn, model_dir='/tmp/multiworker', config=config)
tf.estimator.train_and_evaluate(
    classifier,
    train_spec=tf.estimator.TrainSpec(input_fn=input_fn),
    eval_spec=tf.estimator.EvalSpec(input_fn=input_fn)
)
```

## 6. 학습 성능 최적화

우리는 `tf.distribute.Strategy`로 구동되는 모델과 다중 워커를 지원하는 에스티메이터를 얻었습니다. 이제 다음 기술들을 사용하여 다중 워커의 학습 성능을 최적화할 수 있습니다.

* **배치(batch) 크기 늘리기**: 여기에서의 배치 크기는 GPU 당 크기입니다. 일반적으로 GPU 메모리에 맞는 가장 큰 배치 크기를 권장합니다.
* **변수 지정하기**: 가능한 경우 변수를 tf.float로 지정합니다. 
*   **콜렉티브 커뮤니케이션(collective communication) 사용하기**:  `MultiWorkerMirroredStrategy`를 통해 다중 콜렉티브 커뮤니케이션을 구현할 수 있습니다.
    * `RING`은 gRPC를 크로스 호스트(cross-host) 커뮤니케이션 레이어로 사용하는 링 기반의 콜렉티브를 구현합니다.
    
    * `NCCL`는 콜렉티브를 구현하기 위해 [Nvidia's NCCL](https://developer.nvidia.com/nccl)을 사용합니다.
    * `AUTO`는 런타임에 따라 다르게 설정하도록 합니다.

콜렉티브 구현을 위한 최상의 선택은 GPU 수와 종류, 그리고 클러스터의 네트워크 인터커넥트에 따라 달라집니다. 자동으로 설정된 것을 재정의하려면 "MultiWorker MirroredStrategy"의 `communication` 매개 변수에 올바른 값을 지정합니다. 아래는 그 예시입니다:
`communication=tf.distribute.experimental.CollectiveCommunication.NCCL`.


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
