
# [튜토리얼6] TF.Text 사용하기

이번 튜토리얼에서는 유니코드(Unicode)와 토큰화(Tokenization)에 대해서 배우고 텐서플로우의 **TF.Text**를 사용하는 방법을 살펴보겠습니다.

텐서플로우 텍스트는 TensorFlow 2.0과 함께 사용할 수 있는 텍스트 관련 클래스 및 ops를 제공합니다. 라이브러리는 텍스트 기반 모델에 필요한 전처리를 정기적으로 수행할 수 있으며 코어 텐서플로우(core TensorFlow)에서 제공하지 않는 시퀀스 모델링에 유용한 기타 피쳐들을 포함합니다.

텍스트 전처리에 이러한 작업을 사용하면 텐서플로우 그래프에서 작업을 수행할 수 있습니다. 훈련 중의 토큰화가 인퍼런스(inference)에서의 토큰화와 다르거나 전처리 스크립트를 관리하는 것에 대한 걱정을 할 필요는 없습니다.

# 목차
1. 즉시 실행(Eager Execution)
2. 유니코드(Unicode)
3. 토큰화(Tokenization)
    - 3.1 WhitespaceTokenizer
    - 3.2 UnicodeScriptTokenizer
    - 3.3 Unicode split
    - 3.4 오프셋(Offsets)
    - 3.5 TF.Data 예제
4. 기타 텍스트 Ops
    - 4.1 Wordshape
    - 4.2 N-grams & Sliding Window

## 1. 즉시 실행(Eager Execution)

TensorFlow Text에는 TensorFlow 2.0이 필요하며, 즉시 실행 모드 및 그래프 모드와 완벽하게 호환됩니다.


```python
import warnings
warnings.simplefilter('ignore')

!pip install -q tensorflow-text
import tensorflow as tf
import tensorflow_text as text
```

## 2. 유니코드(Unicode)

대부분의 ops는 문자열이 UTF-8로 되어 있을 것이라 생각합니다. 다른 인코딩을 사용하는 경우 코어 텐서플로우 트랜스코드(transcode) op을 사용하여 UTF-8로 변환할 수 있습니다. 또한 입력이 잘못될 수 있는 경우 동일한 op을 사용하여 문자열을 구조적으로 유효한 UTF-8로 변환할 수도 있습니다.


```python
docs = tf.constant([u'Everything not saved will be lost.'.encode('UTF-16-BE'), u'Sad☹'.encode('UTF-16-BE')])
utf8_docs = tf.strings.unicode_transcode(docs, input_encoding='UTF-16-BE', output_encoding='UTF-8')
```

## 3. 토큰화(Tokenization)

토큰화는 문자열을 토큰으로 분할하는 프로세스입니다. 일반적으로 이러한 토큰은 단어, 숫자 또는 구두점으로 구성되어 있습니다.

주요 인터페이스는 `Tokenizer`와 `TokenizerWithOffsets`로 각각 `tokenize`와 `tokenize_with_offsets`의 단일 메서드가 있습니다. 현재 여러 토크나이저를 사용할 수 있습니다. 

각 토크나이저들은 바이트 오프셋(byte offsets)을 원래 문자열로 가져오는 옵션이 포함된 `TokenizerWithOffsets`를 구현합니다. 이를 통해 호출자(caller)는 토큰이 생성된 원래 문자열의 바이트를 알 수 있습니다. 

모든 토크나이저는 원래의 개별 문자열과 매핑되는 가장 높은 차수를 가진 토큰의 RaggedTensors를 반환합니다. 

결과적으로, 결과 형태(shape)의 순위가 1씩 증가합니다.




### 3.1 WhitespaceTokenizer

이것은 ICU(International Components for Unicode)가 정의한 공백 문자(예: 공백, 탭, 새 줄)로 UTF-8 문자열을 분할하는 기본 토크나이저입니다.


```python
tokenizer = text.WhitespaceTokenizer()
tokens = tokenizer.tokenize(['everything not saved will be lost.', u'Sad☹'.encode('UTF-8')])
print(tokens.to_list())
```

### 3.2 UnicodeScriptTokenizer

이 토크나이저는 유니코드 스크립트 경계(boundaries)를 기준으로 UTF-8 문자열을 분할합니다. 사용된 스크립트 코드(script code)는 ICU UScriptCode의 값(value)에 해당합니다.

실제로 이는 `WhitespaceTokenizer`와 유사합니다. 두 토크나이저의 가장 뚜렷한 차이점은 `UnicodeScriptTokenizer`는 
언어 텍스트(예: USCRIPT_LATIN, USCRIPT_CYRILLIC 등)를 서로 분리하는 동시에 문장 부호(USCRIPT_COMMON)와 언어 텍스트를 분리한다는 것입니다.


```python
tokenizer = text.UnicodeScriptTokenizer()
tokens = tokenizer.tokenize(['everything not saved will be lost.', u'Sad☹'.encode('UTF-8')])
print(tokens.to_list())
```

### 3.3 Unicode split

분할할 단어를 공백 없이 토큰화하고자 할 때는 문자별로 분할하는 것이 일반적이며, 이는 코어(core)에 있는 [unicode_split](https://www.tensorflow.org/api_docs/python/tf/strings/unicode_split) op를 통해 수행할 수 있습니다.


```python
tokens = tf.strings.unicode_split([u"仅今年前".encode('UTF-8')], 'UTF-8')
print(tokens.to_list())
```

### 3.4 오프셋(Offsets)

보통 문자열을 토큰화할 때 토큰이 원래 문자열에서에서 존재했던 위치를 알고 싶어합니다. 이러한 이유로 `TokenizerWithOffsets`를 구현하는 각 토크나이저에는 **tokenize_with_offets** 메서드가 있으며, 이 메서드는 토큰과 함께 바이트 오프셋(byte offsets)을 반환합니다. 
* **offset_starts**는 각 토큰이 시작하는 원래 문자열의 바이트를 나열합니다
* **offset_limits**는 각 토큰이 끝나는 바이트를 나열합니다.


```python
tokenizer = text.UnicodeScriptTokenizer()
(tokens, offset_starts, offset_limits) = tokenizer.tokenize_with_offsets(['everything not saved will be lost.', u'Sad☹'.encode('UTF-8')])
print(tokens.to_list())
print(offset_starts.to_list())
print(offset_limits.to_list())
```

### 3.5 TF.Data 예제

토크나이저는 tf.data API에서 예상한대로 작동합니다. 아래에 간단한 예시가 있습니다.


```python
docs = tf.data.Dataset.from_tensor_slices([['Never tell me the odds.'], ["It's a trap!"]])
tokenizer = text.WhitespaceTokenizer()
tokenized_docs = docs.map(lambda x: tokenizer.tokenize(x))
iterator = iter(tokenized_docs)
print(next(iterator).to_list())
print(next(iterator).to_list())
```

## 4. 기타 텍스트 Ops

TF.Text는 다른 유용한 전처리 작업을 패키지화합니다. 아래 몇 가지 사항을 보겠습니다.

### 4.1 Wordshape

일부 자연어 이해 모델에서 사용되는 일반적인 기능은 텍스트 문자열에 특정 속성이 있는지 확인하는 것입니다. 예를 들어 문단을 문장으로 분리하는 모델에는 단어 대문자 또는 구두점 문자가 문자열의 끝에 있는지 확인하는 피쳐가 포함될 수 있습니다.

Wordshape는 입력 텍스트를 다양한 관련 패턴들과 매칭시키기 위한 다양한 정규식 기반 헬퍼(helper) 함수를 가지고 있습니다. 

아래 몇 가지 예시들을 보겠습니다.


```python
tokenizer = text.WhitespaceTokenizer()
tokens = tokenizer.tokenize(['Everything not saved will be lost.', u'Sad☹'.encode('UTF-8')])

# 대문자인가?
f1 = text.wordshape(tokens, text.WordShape.HAS_TITLE_CASE)
# 모든 문자가 대문자인가?
f2 = text.wordshape(tokens, text.WordShape.IS_UPPERCASE)
# 토큰이 구두점을 포함하는가?
f3 = text.wordshape(tokens, text.WordShape.HAS_SOME_PUNCT_OR_SYMBOL)
# 토큰이 숫자인가?
f4 = text.wordshape(tokens, text.WordShape.IS_NUMERIC_VALUE)

print(f1.to_list())
print(f2.to_list())
print(f3.to_list())
print(f4.to_list())
```

### 4.2 N-grams & Sliding Window

N-그램은 슬라이딩(sliding) 윈도우 크기가 **n**인 시퀀셜(sequential) 단어입니다. 토큰을 결합할 경우 세 가지 감소 메커니즘이 지원됩니다. 텍스트의 경우 `Reduction.STRING_JOIN`를 사용하여 서로의 문자열을 추가합니다. 기본 구분 문자는 공백이지만 `string_separater` 인수를 사용하여 변경할 수 있습니다.

다른 두 가지 감소 방법인 `Reduction.SUM`과 `Reduction.MEAN`는 숫자형 데이터를 다룰 때 많이 사용됩니다.


```python
tokenizer = text.WhitespaceTokenizer()
tokens = tokenizer.tokenize(['Everything not saved will be lost.', u'Sad☹'.encode('UTF-8')])

# Ngrams입니다. 이 경우에는 이진 그램입니다.(n = 2)
bigrams = text.ngrams(tokens, 2, reduction_type=text.Reduction.STRING_JOIN)

print(bigrams.to_list())
```

# Copyright 2018 The TensorFlow Authors.

Licensed under the Apache License, Version 2.0 (the "License");


```python
#@title Licensed under the Apache License, Version 2.0 (the "License"); { display-mode: "form" }
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
