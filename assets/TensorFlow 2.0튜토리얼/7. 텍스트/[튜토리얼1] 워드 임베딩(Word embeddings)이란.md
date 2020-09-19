
# [튜토리얼1] 워드 임베딩(Word embeddings)이란

이 튜토리얼에서는 워드 임베딩에 대해 알아보겠습니다. 이번 튜토리얼은 작은 데이터셋에  워드 임베딩을 처음부터 학습시키고 [Embedding Projector](http://projector.tensorflow.org)를 통해 임베딩한 것을 시각화하는 방법을 보겠습니다. (아래 그림 참조).

<img src="https://github.com/tensorflow/docs/blob/master/site/en/tutorials/text/images/embedding.jpg?raw=1" alt="Screenshot of the embedding projector" width="400"/>



```python
import warnings
warnings.simplefilter('ignore')

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_datasets as tfds
tfds.disable_progress_bar()
```

# 목차
1. 텍스트를 숫자로 나타내기
    - 1.1 원-핫 인코딩(One-hot encodings)
    - 1.2 각 단어를 고유 번호로 인코딩하기
    - 1.3 워드 임베딩(Word embeddings)
2. 임베딩 레이어(Embedding layer) 사용하기
3. 임베딩 처음부터 해보기
    - 3.1 간단한 모델 만들기
    - 3.2 모델을 컴파일하고 학습시키기
4. 학습된 임베딩 찾기
5. 임베딩을 시각화하기

## 1. 텍스트를 숫자로 나타내기

머신러닝 모델은 벡터(숫자 배열)를 입력으로 사용합니다. 텍스트를 다룰 때 먼저 문자열을 숫자로 변환(또는 텍스트를 "벡터화"하기 위해)하는 해야 합니다. 이 섹션에서는 이를 위한 세 가지 방법을 살펴보겠습니다.



### 1.1 원-핫 인코딩(One-hot encodings)
첫 번째 방법은 어휘에 있는 각각의 단어들을 
**원-핫 인코딩**하는 것입니다. "The cat sat on the mat"라는 문장을 예로 들어봅시다. 이 문장에서의 어휘(또는 고유 단어)는 (cat, mat, on, sat, the)입니다. 각 단어를 나타내기 위해, 어휘의 개수와 같은 길이를 가진 영벡터(zero vector)를 만든 다음, 단어와 일치하는 인덱스에 1을 입력합니다. 이 접근 방식은 아래의 다이어그램에 나와 있습니다.

<img src="https://github.com/tensorflow/docs/blob/master/site/en/tutorials/text/images/one-hot.png?raw=1" alt="Diagram of one-hot encodings" width="400" />

각 단어에 대한 원-핫 벡터를 연결하여 문장의 인코딩 벡터를 만듭니다.

키 포인트: 이 방법은 비효율적입니다. 원-핫 인코딩 벡터는 희소(sparse)합니다(즉, 대부분의 표시는 0임). 어휘 사전(vocabulary)에 10,000개의 단어가 있다고 생각해봅시다. 이 경우 각 단어에 해당하는 1개의 열을 인코딩하기 위해 99.99%의 요소가 0인 벡터를 만듭니다.

### 1.2 각 단어를 고유 번호로 인코딩하기

두 번째 방법은 고유 번호를 사용하여 각 단어를 인코딩하는 것입니다. 위의 예제를 다시 사용하면 "cat"에 1을, "mat"에 2를 할당하는 등 전체 단어에 할당할 수 있습니다. 그런 다음 "The cat sat on the mat"라는 문장을 [5, 1, 4, 3, 5, 2]와 같은 밀도 있는 벡터로 부호화할 수 있었습니다. 이 방법으로는 희소 벡터가 아닌 밀도가 높은 벡터(모든 요소가 가득 찬 벡터)를 갖기 때문에 효율적입니다.

그러나 이 방식에는 두 가지 단점이 있습니다.

* 정수 인코딩은 독단적(arbitrary)입니다(단어 간의 관계를 반영하지 않음).

* 정수 인코딩은 모델을 해석하기 어려울 수 있습니다. 예를 들어 선형 분류기는 각 피쳐에 대한 단일 가중치를 학습합니다. 두 단어의 유사성과 인코딩의 유사성 사이에는 관계가 없으므로 이 피쳐와 가중치의 조합은 의미가 없습니다.

### 1.3 워드 임베딩(Word embeddings)

워드 임베딩은 유사한 단어끼리 유사하게 인코딩되는 효율적이고 밀도 있게 표현하는 방법입니다. 중요한 것은 이 인코딩을 직접 지정할 필요가 없다는 것입니다. 임베딩은 부동 소수점 값(floating point)을 가진 밀도 벡터(dense vector)입니다. 이 때 벡터의 길이는 매개변수이므로 지정해줘야 합니다. 임베딩 값은 직접 지정하지 않고 훈련 가능한 매개변수로서 밀도(dense) 레이어로 가중치를 학습하는 것과 같은 방법으로 모델을 학습시켜서 나온 가중치로 임베딩합니다. 일반적으로 큰 데이터셋을 작업할 때에는 작게는 8차원에서 최대 1024차원으로 워드 임베딩을 합니다. 높은 차원의 임베딩일수록 단어 간의 세부적인 관계를 파악할 수 있지만 학습하는 데 더 많은 데이터를 필요로 합니다.

<img src="https://github.com/tensorflow/docs/blob/master/site/en/tutorials/text/images/embedding2.png?raw=1" alt="Diagram of an embedding" width="400"/>

위 그림은 워드 임베딩의 다이어그램입니다. 각 단어는 부동 소수점(floating point) 값의 4차원 벡터로 표시됩니다. 임베딩한 것을 확인하는하는 또 다른 방법은 "룩업(lookup) 테이블"입니다. 위의 과정에서 가중치들이 학습되고 난 후, 룩업 테이블에 해당하는 밀도 벡터를 찾아 각 단어를 인코딩할 수 있습니다.

## 2. 임베딩 레이어(Embedding layer) 사용하기

케라스는 워드 임베딩을 쉽게 사용할 수 있도록 해줍니다. [임베딩](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding) 레이어를 살펴보겠습니다.

임베딩 레이어(Embedding layer)는 특정 단어를 나타내는 정수 인덱스를 이 밀도 벡터와 매핑되는 룩업 테이블로 이해할 수 있습니다. Dense 레이어에서 뉴런의 수를 실험하는 것처럼 임베딩의 차원수(또는 너비)도 무엇이 상황에 잘 맞는지 알아보기 위해 실험해볼 수 있는 매개변수입니다.




```python
embedding_layer = layers.Embedding(1000, 5)
```

임베딩 레이어을 만들면 임베딩에 대한 가중치는 다른 레이어들과 마찬가지로 임의로 초기화됩니다. 모델이 학습하면서 오차역전파(backpropagation)를 통해 가중치를 점진적으로 조정됩니다. 일단 학습하게 되면, 학습된 워드 임베딩은 모델이 학습한 특정 문제에 대해 학습한 것처럼 단어 간의 유사성을 대략적으로 인코딩합니다.

정수를 임베딩 레이어에 전달하면 임베딩 테이블에서 각 정수를 벡터로 바꿉니다.


```python
result = embedding_layer(tf.constant([1,2,3]))
result.numpy()
```

텍스트 또는 시퀀스 문제의 경우, 임베딩 레이어는 `(sample, sequence_length)` 형태인 2D 정수 텐서를 사용하며, 여기서 각 입력은 정수 값을 가진 시퀀스입니다. 가변적인 길이의 시퀀스도 입력값으로 사용할 수 있습니다. 예를 들어, 배치 위의 임베딩 레이어에 `(32, 10)`(길이 10의 32 시퀀스 배치)나 `(64, 15)`(길이 15의 64 시퀀스 배치) 형태를 입력할 수 있습니다.

반환된 텐서는 입력보다 하나의 축이 더 있으며, 임베딩 벡터는 새로 생기는 마지막 축에 따라 정렬됩니다. `(2, 3)`의 입력 배치를 전달하면 출력값은 `(2, 3, N)`입니다.


```python
result = embedding_layer(tf.constant([[0,1,2],[3,4,5]]))
result.shape
```

입력으로 시퀀스 배치를 지정하면 임베딩 레이어는 `(samples, sequence_length, embedding_dimensionality)` 형태의 3D 부동 소수점(floating point) 텐서를 반환합니다. 이런 가변적인 길이의 시퀀스를 고정적인 형태로 변환하는 방법은 다양합니다. Dense 레이어로 전달하기 전에 RNN이나 어텐션(attention), 또는 풀링(pooling) 레이어를 사용하면 됩니다. 이 튜토리얼은 가장 간단한 풀링을 사용합니다. 

## 3. 임베딩 처음부터 해보기

이 튜토리얼에서는 IMDB 동영상 리뷰에 대한 감성 분류기를 학습합니다. 이 과정에서 모델은 처음부터 임베딩을 학습합니다. 전처리된 데이터셋에 사용할 것입니다.


```python
(train_data, test_data), info = tfds.load(
    'imdb_reviews/subwords8k', 
    split = (tfds.Split.TRAIN, tfds.Split.TEST), 
    with_info=True, as_supervised=True)
```

인코더(`tfds.features.text.SubwordTextEncoder`)를 가져와 어휘 사전(vocabulary)을 간략히 살펴봅시다.

어휘 사전에서 "\_"는 공백을 나타냅니다. 어휘 사전이 "\_"로 끝나는 단어와 더 큰 단어를 만드는 데 사용할 수 있는 부분 단어를 어떻게 포함하고 있는지를 확인해봅시다.


```python
encoder = info.features['text'].encoder
encoder.subwords[:20]
```

영화 리뷰는 길이가 다를 것입니다. 리뷰의 길이를 표준화하기 위해 `padded_batch` 메서드를 사용합니다.


```python
train_batches = train_data.shuffle(1000).padded_batch(10, ((None,),()))
test_batches = test_data.shuffle(1000).padded_batch(10, ((None,),()))
```

임포트 된대로 리뷰 텍스트는 정수 인코딩됩니다(각 정수는 어휘 사전에서 특정 단어 또는 단어 부분을 나타냄).

배치가 가장 긴 예제로 패딩되기 때문에 뒤에는 0으로 만듭니다.


```python
train_batch, train_labels = next(iter(train_batches))
train_batch.numpy()
```

### 3.1 간단한 모델 만들기
[Keras Sequential API](../../guide/keras)을 사용하여 모델을 정의합니다.이 경우 "Continuous bag of words" 스타일의 모델을 사용합니다.

* 다음으로 임베딩 레이어는 정수 인코딩된 어휘 사전을 사용하고 각 워드 인덱스에 대한 임베딩 벡터를 찾습니다. 이 벡터들은 모델 학습함에따라 함께 학습됩니다. 벡터는 출력 배열에 차원수를 추가합니다. 그 결과 차원수는 `(batch, sequence, embedding)`의 형태가 됩니다.

* 그런 다음 GlobalAveragePooling1D 레이어는 시퀀스 차원의 평균을 계산하여 각 샘플에 대해 고정된 길이를 가진 출력 벡터를 반환합니다. 이를 통해 모델은 가장 간단한 방법으로 가변적인 길이를 가진 입력값을 처리할 수 있습니다.

* 이렇게 고정된 길이의 출력 벡터는 16개의 은닉 유닛(unit)이 있는 완전히 연결된(Dense) 레이어에 연결됩니다.

* 마지막 레이어는 단일 출력 노드와 밀접하게 연결됩니다. sigmoid 활성화 함수를 사용하여 이 값은 0과 1 사이의 float 값으로, 리뷰가 긍정일 확률(또는 신뢰 수준)을 나타냅니다.

주의: 이 모델은 마스킹을 사용하지 않기 때문에 0으로 패딩을 만든 것이 입력의 일부로 들어가서 패딩 길이가 출력에 영향을 줄 수 있습니다.


```python
embedding_dim=16

model = keras.Sequential([
  layers.Embedding(encoder.vocab_size, embedding_dim),
  layers.GlobalAveragePooling1D(),
  layers.Dense(16, activation='relu'),
  layers.Dense(1, activation='sigmoid')
])

model.summary()
```

### 3.2 모델을 컴파일하고 학습시키기


```python
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_batches,
    epochs=10,
    validation_data=test_batches, validation_steps=20)
```

이 방법으로 모델은 85% 이상의 정확도를 가집니다.(하지만 이 모델은 학습 정확도가 현저히 높으므로 이는 오버피팅 된 것입니다.)


```python
import matplotlib.pyplot as plt

history_dict = history.history

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.figure(figsize=(12,9))
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure(figsize=(12,9))
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim((0.5,1))
plt.show()
```

## 4. 학습된 임베딩 찾기

다음은 학습하면서 배운 워드 임베딩한 것을 찾아보겠습니다. 워드 임베딩은 `(어휘 사전의 크기, 임베딩의 차원)`의 형태인 행렬이 될 것입니다.


```python
e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape) # shape: (vocab_size, embedding_dim)
```

이제 디스크에 가중치를 기록하겠습니다. [Embedding Projector](http://projector.tensorflow.org)를 사용하기 위해 임베딩을 포함하는 벡터 파일과 단어를 포함하는 메타 데이터 파일을 TSV 파일로 업로드합니다.


```python
import io

encoder = info.features['text'].encoder

out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')

for num, word in enumerate(encoder.subwords):
    vec = weights[num+1] # skip 0, it's padding.
    out_m.write(word + "\n")
    out_v.write('\t'.join([str(x) for x in vec]) + "\n")
out_v.close()
out_m.close()
```

## 5. 임베딩을 시각화하기

임베딩한 것을 시각화하려면 임베딩 프로젝터(Embedding Projector)에 이를 업로드합니다.

임베딩 프로젝터를 엽니다.

* "Load data"를 클릭합니다.

* 위에서 만든 두 파일 `vecs.tsv`와 `meta.tsv`를 업로드합니다.

그러면 학습한 임베딩이 나타납니다. 이제 단어를 검색해서 그 단어의 가장 가까운 이웃(neighbor)을 찾을 수 있습니다. 예를 들어, "아름답다"를 검색해보면 "멋지다"와 같은 이웃들을 볼 수 있습니다.

참고: 임베딩 레이어를 학습하기 전에 가중치가 임의로 초기화된 방법에 따라 결과가 약간 다를 수 있습니다.

참고: 더 단순한 모델을 사용하여 좀 더 해석하기 쉬운 임베딩을 생성할 수 있습니다. `Dense(16)` 레이어를 삭제하고 모델을 다시 학습시킨 다음 임베딩을 다시 시각화해보세요.

<img src="https://github.com/tensorflow/docs/blob/master/site/en/tutorials/text/images/embedding.jpg?raw=1" alt="Screenshot of the embedding projector" width="400"/>


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
