
# [튜토리얼8] TFRecord와 tf.Example

이번 튜토리얼에서는 TFRecord와 tf.Example을 다루는 방법을 보겠습니다.

데이터를 효율적으로 읽으려면 데이터를 나열하고 연속적으로 읽을 수 있도록 파일 집합(각 100-200MB)에 저장하는 것이 유용합니다. 이는 특히 데이터를 네트워크를 통해 스트리밍하는 경우에 더욱 효과적입니다. 이 기능은 데이터 전처리를 캐싱하는 데에도 유용할 것입니다.

TFRecord 형식은 이진 레코드의 시퀀스를 저장할 수 있는 간단한 형식입니다.

[프로토콜 버퍼](https://developers.google.com/protocol-buffers/)는 구조화된 데이터의 효율적인 직렬화(serialize)를 위한 교차 플랫폼(cross-platform), 교차 언어(cross-language) 라이브러리입니다.

* 프로토콜 메시지는 `.proto` 파일로 정의되며, 이러한 메시지는 메시지 타입을 이해하는 가장 쉬운 방법입니다.
* `tf.Example` 메시지(또는 protobuf)는 `{"string": value}` 매핑을 나타내는 유연한 메시지 유형입니다.
* TensorFlow와 함께 사용하도록 설계되었으며 TFX와 같은 상위 레벨 API에서 사용됩니다.

이 튜토리얼에서는 `tf.Example` 메세지를 만들고 분석하며 사용하는 방법을 보여줄 것입니다. 그 후  `tf.Example` 메세지를 `.tfrecord`에서 직렬화(serialize)하거나 작성하고 읽어옵니다.

참고: 이 방법은 유용하지만 선택 사항입니다. 이미 [`tf.data`](https://www.tensorflow.org/guide/datasets)를 사용 중이거나 데이터를 읽는 것이 훈련 과정에서의 병목 현상인 경우가 아니라면 굳이 기존 코드에서 TFRecords를 사용하기위해 변환할 필요는 없습니다.


```python
import warnings
warnings.simplefilter('ignore')

import tensorflow as tf
import numpy as np
import IPython.display as display
```

# 목차
1. tf.Example
2. TFRecords의 세부 정보
3. tf.data로 TFRecord 파일 다루기
4. Python에서 TFRecord 파일 다루기
5. 예제: 이미지 데이터를 불러오고 작성하기

## 1. `tf.Example`

### 1.1 `tf.Example`의 데이터 타입

기본적으로 `tf.Example`은 `{"string": tf.train.Feature}` 매핑으로 이루어져 있습니다.

`tf.train.Feature` 메시지 타입은 다음 세 가지 유형 중 하나로 사용할 수 있습니다. 대부분의 다른 제네릭(generic) 타입은 다음 중 하나로 강제 적용할 수 있습니다.

1. `tf.train.BytesList` (다음 유형을 강제 적용할 수 있습니다.)

  - `string`
  - `byte`

2. `tf.train.FloatList` (다음 유형을 강제 적용할 수 있습니다.)

  - `float` (`float32`)
  - `double` (`float64`)

3. `tf.train.Int64List` (다음 유형을 강제 적용할 수 있습니다.)

  - `bool`
  - `enum`
  - `int32`
  - `uint32`
  - `int64`
  - `uint64`

아래의 방법들을 통해 표준 텐서플로우 타입을 `tf.train.Feature`와 호환가능한 `tf.Example`로  손쉽게 변환할 수 있습니다. 

참고로 각 함수는 스칼라 입력 값을 사용하고 위의 세 가지 `list` 타입 중 하나로 되어있는 `tf.train.Feature`을 반환합니다.


```python
# 다음 함수들을 사용하여 값을 tf.Example과 호환되는 형식으로 변환할 수 있습니다

def _bytes_feature(value):
    """string / byte를 byte_list로 반환합니다."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # ByteList는 EagerTensor의 문자열을 언팩(unpack)하지 않습니다.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """float / double을 float_list로 반환합니다."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """bool / enum / int / unit을 int64_list로 반환합니다."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
```

참고: 단순성을 유지하기 위해 이 예시에서는 스칼라 입력만 사용합니다. 스칼라가 아닌 피쳐를 처리하는 가장 간단한 방법은 `tf.serialize_tensor`를 사용하여 텐서를 이진 스트링으로 변환하는 것입니다. 문자열은 텐서플로우에서는 스칼라입니다. `tf.parse_tensor`를 사용하여 이진 문자열을 다시 텐서로 변환합니다.

다음은 이러한 함수의 작동 원리를 보여 주는 몇 가지 예시들을 보여줍니다. 함수에 대한 입력 유형이 위에서 언급한 강제성 유형과 일치하지 않을 경우 예외가 발생합니다(예: `_int64_feature(1.0)`는 `1.0`이 float이기 때문에 에러가 발생할 것입니다. 그러므로 그 대신 `_float_feature`을 사용해야 합니다.):


```python
print(_bytes_feature(b'test_string'))
print(_bytes_feature(u'test_bytes'.encode('utf-8')))

print(_float_feature(np.exp(1)))

print(_int64_feature(True))
print(_int64_feature(1))
```

모든 프로토(proto) 메시지는 `.SerializeToString` 메서드를 사용하여 이진 문자열로 직렬화(serialize)할 수 있습니다.


```python
feature = _float_feature(np.exp(1))

feature.SerializeToString()
```

### 1.2 `tf.Example` 메세지 만들기

기존 데이터에 `tf.Example` 메세지를 생성한다고 해봅시다. 실제로 데이터셋은 다양하게 제공될 수 있지만 단일 관측치에 대해 `tf.Example` 메세지를 생성하는 절차는 동일합니다.

1. 각 관측치 내의 각 값들을 위의 함수들을 사용하여 위 3가지의 호환 가능한 타입을 가지고있는 `tf.train.Feature`로 변환해야 합니다.

2. 피쳐(feature) 이름 문자열에서 1단계에서 생성된 인코딩된 피쳐 값으로 맵(딕셔너리)을 만듭니다.

3. 2단계에서 생성된 맵(map)을 [피쳐 메세지](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/example/feature.proto#L85)(Features message)로 변환합니다.

이번 튜토리얼에서는 NumPy를 사용해 데이터셋을 생성할 것입니다.

이 데이터셋은 아래의 4가지 피쳐로 구성됩니다:

* 동일한 확률을 가진 `False`나 `True`값을 가지는 불리언(boolean) 피쳐
* 균일하게 임의로 선택된 0과 5사이의 정수형(integer) 피쳐
* 정수 피쳐를 인덱스로 사용하는 문자열 테이블에서 생성된 문자열(string) 피쳐
* 표준 정규 분포를 따르는 실수형(float) 피쳐

위의 각 분포에서 10,000개의 독립적이고 동일한 분포의 관측치로 구성된 표본을 사용해봅시다.


```python
# 데이터셋 내 관값의 수
n_observations = int(1e4)

# False나 True로 인코딩하는 불리언 피쳐
feature0 = np.random.choice([False, True], n_observations)

# 0과 5사이의 무작위 정수를 뽑는 정수형 피쳐
feature1 = np.random.randint(0, 5, n_observations)

# 문자열 피쳐
strings = np.array([b'cat', b'dog', b'chicken', b'horse', b'goat'])
feature2 = strings[feature1]

# 표준 정규화 분포를 따르는 실수형 피쳐
feature3 = np.random.randn(n_observations)
```

각 피쳐들은 강제로 `_bytes_feature`, `_float_feature`, `_int64_feature` 중 하나를 사용해서 호환가능한 `tf.Example`으로 만들 수 있습니다. 이후 위 방법으로 인코딩된 피쳐들을 이용해 `tf.Example` 메세지를 생성할 수 있습니다.


```python
def serialize_example(feature0, feature1, feature2, feature3):
    """
    Creates a tf.Example message ready to be written to a file.
    """
    # 피쳐 이름을 tf.Example-compatible의 데이터 타입과 맞게 매핑한 사전을 만듭니다.
    feature = {
        'feature0': _int64_feature(feature0),
        'feature1': _int64_feature(feature1),
        'feature2': _bytes_feature(feature2),
        'feature3': _float_feature(feature3),
    }

    # tf.train.Example을 이용해 Features 메시지를 만듭니다
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()
```

예를 들어, 데이터셋에서 다음과 같은 단일 관측치가 있다고 가정합니다. 
* `[False, 4, bytes('goat'), 0.9876]`

`create_message()`는 이 관측치에 관한 `tf.Example` 메세지를 생성하고 출력할 수 있도록 해줍니다. 각 단일 관측치는 위와 같이 `Features` 메시지로 기록됩니다. 

여기서 `tf.Example` 메세지는 단지 `Features` 메세지에 대한 래퍼(wrapper)일 뿐입니다.


```python
# 데이터셋의 관측치를 보겠습니다.

example_observation = []

serialized_example = serialize_example(False, 4, b'goat', 0.9876)
serialized_example
```

`tf.train.Example.FromString`를 이용하여 메세지를 디코딩할 수 있습니다.


```python
example_proto = tf.train.Example.FromString(serialized_example)
example_proto
```

## 2. TFRecords의 세부 정보

TFRecord 파일에는 일련의 레코드(record)가 포함되어 있습니다. 파일은 순차적으로만 읽을 수 있습니다.

각 레코드에는 데이터 페이로드(data-payload)에 대한 바이트 문자열과 데이터 길이 및 무결성(integrity) 검사를 위한 CRC32C(Castagnoli 다항식을 사용하는 32비트 CRC) 해시(hash)가 포함되어 있습니다.

각 레코드는 다음과 같은 형식으로 저장됩니다:

    uint64 length
    uint32 masked_crc32_of_length
    byte   data[length]
    uint32 masked_crc32_of_data

레코드는 파일을 만들기 위해 함께 연결되어 있습니다. CRC의 마스크(mask)는 다음과 같습니다:

    masked_crc = ((crc >> 15) | (crc << 17)) + 0xa282ead8ul

참고: `tf.Example`은 TFRecord 파일에서는 사용할 필요가 없습니다. `tf.Example`은 `tf.io.serialize_tensor`와
`tf.io.parse_tensor`로 불러온 텍스트 라인, 인코딩된 이미지 데이터 또는 직렬화된 텐서와 같은 바이트 문자열을 순서대로 나열하는 디렉토리 메서드일 뿐입니다.

## 3. `tf.data`로 TFRecord 파일 다루기

`tf.data` 모듈은 TensorFlow에서 데이터를 읽고 작성하는 도구도 제공합니다.

### 3.1 TFRecord 파일 작성하기

데이터를 데이터셋으로 가져오는 가장 쉬운 방법은 `from_tensor_slices` 메서드를 사용하는 것입니다.

배열에 적용하면 다음과 같은 스칼라 데이터셋을 반환합니다.


```python
tf.data.Dataset.from_tensor_slices(feature1)
```

튜플 배열에 적용하면 튜플 데이터셋을 반환합니다.


```python
features_dataset = tf.data.Dataset.from_tensor_slices((feature0, feature1, feature2, feature3))
features_dataset
```


```python
# 데이터셋에서 하나의 데이터만을 추출하려면 `take(1)`를 사용합니다.
for f0,f1,f2,f3 in features_dataset.take(1):
    print(f0)
    print(f1)
    print(f2)
    print(f3)
```

`tf.data.Dataset.map` 메서드를 사용하면 `Dataset`의 각 요소에 함수를 적용하는 할 수 있습니다.

매핑된 함수는 TensorFlow 그래프 모드에서 실행되어야 하며, `tf.Tensors`로 실행하고 반환해야 합니다. `serialize_example`과 같은 텐서가 아닌 함수는 `tf.py_function`으로 감싸서 호환되도록 만들 수 있습니다.

`tf.py_function`을 사용할 때에는 모양 및 입력 정보를 지정해야 합니다. 그렇지 않으면 사용할 수 없습니다:


```python
def tf_serialize_example(f0,f1,f2,f3):
    tf_string = tf.py_function(
        serialize_example,
        (f0,f1,f2,f3),  # 함수에 이 인자들을 전달합니다.
        tf.string)      # 반환되는 데이터 타입은 `tf.string`입니다.
    return tf.reshape(tf_string, ()) # 결과물은 스칼라 데이터입니다.
```


```python
tf_serialize_example(f0,f1,f2,f3)
```

데이터셋의 각 요소에 이 함수를 적용하세요:


```python
serialized_features_dataset = features_dataset.map(tf_serialize_example)
serialized_features_dataset
```


```python
def generator():
    for features in features_dataset:
        yield serialize_example(*features)
```


```python
serialized_features_dataset = tf.data.Dataset.from_generator(
    generator, output_types=tf.string, output_shapes=())
```


```python
serialized_features_dataset
```

그리고 이것을 TFRecord 파일에 작성합니다:


```python
filename = 'test.tfrecord'
writer = tf.data.experimental.TFRecordWriter(filename)
writer.write(serialized_features_dataset)
```

### 3.2 TFRecord 파일 읽어오기

`tf.data.TFRecordDataset` 클래스로도 TFRecord 파일을 읽어올 수 있습니다.

`TFRecordDataset`은 입력 데이터를 표준화하고 성능을 최적화하는 데 유용합니다.


```python
filenames = [filename]
raw_dataset = tf.data.TFRecordDataset(filenames)
raw_dataset
```

이 때 데이터셋에는 직렬화된 `tf.train.Example` 메세지가 포함되어 있습니다. 이를 데이터셋에 전부 반복(iterate)하면 스칼라 문자열 텐서로 반환됩니다.

`.take` 메서드를 사용하면 처음 10개의 레코드만을 보여줍니다.

참고: `tf.data.Dataset`에 대해서 반복(iterate)하는 것은 즉시 실행 모드에서만 가능합니다.


```python
for raw_record in raw_dataset.take(10):
    print(repr(raw_record))
```

위 텐서들은 아래 함수를 통해 구문 분석할 수 있습니다. 여기에서는 데이터셋이 그래프 실행(graph-execution)을 사용하고 데이터셋의 모양(shape)과 타입 시그니처(type signature)를 생성하기 위해서는 이에 대한 정보가 필요하기 때문에 `feature_description`을 사용해야 합니다.


```python
# 피쳐들의 간단한 정보를 보여주는 description을 만듭니다
feature_description = {
    'feature0': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'feature1': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'feature2': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'feature3': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
}

def _parse_function(example_proto):
  # 위의 딕셔너리를 이용해서 입력값인 `tf.Example` 프로토를 파싱합니다.
  return tf.io.parse_single_example(example_proto, feature_description)
```

또는 `tf.parse example`를 사용하여 전체 배치를 한 번에 구문 분석할 수 있습니다. 이 함수를 `tf.data.Dataset.map` 메서드로 데이터셋의 각 항목에 적용합니다.


```python
parsed_dataset = raw_dataset.map(_parse_function)
parsed_dataset
```

즉시 실행을 사용하여 데이터 세트에 관측치를 봅시다. 이 데이터셋에는 10,000개의 관측치가 있지만 우선 처음 10개만 보겠습니다. 
* 데이터는 피쳐들의 딕셔너리로 표시됩니다. 
* 각 항목은 `tf.Tensor`이고 이 텐서의 `numpy` 요소는 피쳐 값을 보여줍니다.


```python
for parsed_record in parsed_dataset.take(10):
    print(repr(parsed_record))
```

여기서 `tf.parse_example` 함수는 `tf.Example` 필드를 표준 텐서로 변환해줍니다.

## 4. Python에서 TFRecord 파일 다루기

`tf.io` 모듈에는 TFRecord 파일을 읽고 쓸 수 있는 순수 파이썬 함수들도 포함되어 있습니다.

### 4.1 TFRecord 파일 작성하기

그런 다음 10,000개의 관측치를 `test.tfrecord` 파일에 작성합니다. 각 관측치를 `tf.Example` 메세지로 변환하여 파일에 작성합니다. 작성한 후에야 `test.tfrecord` 파일이 생성되었는지 확인할 수 있습니다:


```python
# `tf.Example` 관측치들을 파일에 작성합니다.
with tf.io.TFRecordWriter(filename) as writer:
    for i in range(n_observations):
        example = serialize_example(feature0[i], feature1[i], feature2[i], feature3[i])
        writer.write(example)
```


```python
!du -sh {filename}
```

### 4.2 TFRecord 파일 읽어오기

이러한 직렬화된 텐서는 `tf.train.Example.ParseFromString`을 이용하여 쉽게 파싱할 수 있습니다:


```python
filenames = [filename]
raw_dataset = tf.data.TFRecordDataset(filenames)
raw_dataset
```


```python
for raw_record in raw_dataset.take(1):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    print(example)
```

## 5. 예제: 이미지 데이터를 불러오고 작성하기

이번에는 TFRecords를 사용하여 이미지 데이터를 읽고 쓰는 방법을 보여 주는 엔드 투 엔드(end-to-end)에 대한 예제를 보겠습니다. 이미지를 입력 데이터로 받으면 이를 TFRecord 파일로 작성한 다음 파일을 다시 읽고 이미지를 표시합니다.

이는 동일한 입력 데이터셋에서 여러 모델을 사용하려는 경우 유용합니다. 이미지 데이터를 raw 데이터 상태로 저장하지 않고 TFRecord 형식으로 전처리할 수 있고 이 방법으로 모든 추가적인 전처리나 모델링에 사용할 수 있습니다.

먼저 눈 속에 있는 고양이의 [이미지](https://commons.wikimedia.org/wiki/File:Felis_catus-cat_on_snow.jpg)와 건설중인 뉴욕 윌리엄스버그 다리 [사진](https://upload.wikimedia.org/wikipedia/commons/f/fe/New_East_River_Bridge_from_Brooklyn_det.4a09796u.jpg)을 다운로드해 보겠습니다.

### 5.1 이미지 가져오기


```python
cat_in_snow  = tf.keras.utils.get_file('320px-Felis_catus-cat_on_snow.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/320px-Felis_catus-cat_on_snow.jpg')
williamsburg_bridge = tf.keras.utils.get_file('194px-New_East_River_Bridge_from_Brooklyn_det.4a09796u.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/194px-New_East_River_Bridge_from_Brooklyn_det.4a09796u.jpg')
```


```python
display.display(display.Image(filename=cat_in_snow))
display.display(display.HTML('Image cc-by: <a "href=https://commons.wikimedia.org/wiki/File:Felis_catus-cat_on_snow.jpg">Von.grzanka</a>'))
```


```python
display.display(display.Image(filename=williamsburg_bridge))
display.display(display.HTML('<a "href=https://commons.wikimedia.org/wiki/File:New_East_River_Bridge_from_Brooklyn_det.4a09796u.jpg">From Wikimedia</a>'))
```

### 5.2 TFRecord 파일 작성하기

이전과 마찬가지로 `tf.Example`과 호환되는 타입으로 피쳐를 인코딩합니다. raw 이미지 문자열 피쳐는 물론 높이, 너비, 깊이 및 임의의 '라벨' 피쳐도 저장합니다. 파일을 작성할 때 맨 끝부분에 위치한 피쳐는 고양이 이미지와 다리 이미지를 구분하기 위해  사용합니다. 
* 고양이 이미지에는 0, 다리 이미지에는 1을 사용합니다.


```python
image_labels = {
    cat_in_snow : 0,
    williamsburg_bridge : 1,
}
```


```python
# 고양이 이미지를 사용하는 예시입니다.
image_string = open(cat_in_snow, 'rb').read()

label = image_labels[cat_in_snow]

# 관련이 있을 수도 있는 피쳐들의 사전을 만듭니다.
def image_example(image_string, label):
    image_shape = tf.image.decode_jpeg(image_string).shape

    feature = {
        'height': _int64_feature(image_shape[0]),
        'width': _int64_feature(image_shape[1]),
        'depth': _int64_feature(image_shape[2]),
        'label': _int64_feature(label),
        'image_raw': _bytes_feature(image_string),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))

for line in str(image_example(image_string, label)).split('\n')[:15]:
    print(line)
print('...')
```

이제 모든 피쳐들이 `tf.Example` 메세지에 저장됩니다. 그런 다음 위의 코드를 작동시키고 `images.tfrecords` 파일에 예시의 메세지를 기록합니다.


```python
# raw 이미지 파일을 `images.tfrecords`에 작성합니다.
# 먼저 두 이미지를 `tf.Example` 메세지로 처리합니다.
# 그리고 `.tfrecords` 파일로 작성합니다.
record_file = 'images.tfrecords'
with tf.io.TFRecordWriter(record_file) as writer:
    for filename, label in image_labels.items():
        image_string = open(filename, 'rb').read()
        tf_example = image_example(image_string, label)
        writer.write(tf_example.SerializeToString())
```


```python
!du -sh {record_file}
```

### 5.3 TFRecord 파일 읽어오기

이제 `images.tfrecords` 파일이 생성되었으며, 해당 파일에 있는 레코드를 반복하여 작성한 내용을 다시 읽을 수 있습니다. 이 예제에서는 이미지만 재생성할 수 있으므로 raw 이미지 문자열 피쳐만 있으면 됩니다. 
* 위에서 설명한 방법인 `example.features.feature['image_raw'].bytes_list.value[0]`으로 이를 추출합니다. 
* 레이블을 사용하여 어떤 레코드가 고양이이고 어떤 레코드가 다리인지도 확인할 수 있습니다


```python
raw_image_dataset = tf.data.TFRecordDataset('images.tfrecords')

# 피쳐들을 설명하는 사전을 생성합니다.
image_feature_description = {
    'height': tf.io.FixedLenFeature([], tf.int64),
    'width': tf.io.FixedLenFeature([], tf.int64),
    'depth': tf.io.FixedLenFeature([], tf.int64),
    'label': tf.io.FixedLenFeature([], tf.int64),
    'image_raw': tf.io.FixedLenFeature([], tf.string),
}

def _parse_image_function(example_proto):
  # 위의 사전을 이용하여 입력 tf.Example 프로토를 구문 분석합니다.
  return tf.io.parse_single_example(example_proto, image_feature_description)

parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
parsed_image_dataset
```

TFRecord 파일에서 이미지를 복구합니다.


```python
for image_features in parsed_image_dataset:
    image_raw = image_features['image_raw'].numpy()
    display.display(display.Image(data=image_raw))
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
