
# [튜토리얼 6] tf.function으로 성능 향상시키기

텐서플로우 2.0에서는 기본적으로 즉시 실행(eager excution)이 켜져 있습니다. 사용자 인터페이스는 직관적이고 유연합니다(단일 작업 실행이 훨씬 쉽고 빠릅니다). 그러나 이는 성능과 전개성(deployability)을 저하시킬 수 있습니다.

최고의 성능을 얻고 어디에서나 모델을 배포할 수 있도록 하려면 프로그램 내에서 그래프를 만드는  `tf.function`을 사용해야 합니다.
AutoGraph 덕분에 많은 양의 파이썬 코드가 tf.function으로 작동됩니다. 하지만 여전히 조심해야 할 사항이 있습니다.

권장 사항은 다음과 같습니다.

- 개체 변이(object mutation) 또는 리스트에 추가하는 것과 같은 파이썬 부작용도(side effect)에 의존하지 마세요.
- tf.function을 사용할 때에는 넘파이 ops나 Python primitive보다 TensorFlow ops가 가장 좋습니다.
- 의심스러운 경우 `for x in y`로 확인해보세요.


```python
import warnings
warnings.simplefilter('ignore')

import tensorflow as tf
```

# 목차
1. 기본 사항
2. 추적(Tracing)과 다형성(Polymorphism)
3. 언제 추적을 할까?
4. 파이썬 arg 또는 텐서 args?
5. tf.function의 부작용도(Side effect)
6. 파이썬 상태(state)를 주의하기
7. 자동 제어 종속변수
8. 변수
9. 오토그래프(AutoGraph) 사용하기
10. 오토그래프: 조건문
11. 오토그래프와 반복문
    - 11.1 For문
    - 11.2 Gotcha's
        - 제로 반복자(Zero iterations)
        - 일관성 있는 형태와 타입

다음과 같은 유형의 오류가 발생할 수 있음을 보여 주는 헬퍼(helper) 함수를 정의합니다:


```python
import traceback
import contextlib

# 일부 헬퍼 코드는 발생할 수 있는 오류의 종류를 보여 줍니다.
@contextlib.contextmanager
def assert_raises(error_class):
    try:
        yield
    except error_class as e:
        print('예상치 못한 예외가 발견되었습니다 \n  {}:'.format(error_class))
        traceback.print_exc(limit=2)
    except Exception as e:
        raise e
    else:
        raise Exception('{}는 에러가 발생할 것이라 예상했지만 발생하지 않았습니다!'.format(
            error_class))
```

## 1. 기본 사항

`tf.function`은 코어 텐서플로우 작업과 똑같습니다. 예를 들어:
* 즉시 실행을 할 수 있습니다.
* 그래프 모드를 사용할 수 있습니다.
* 그래디언트(gradient)가 있습니다.


```python
@tf.function
def add(a, b):
    return a + b

add(tf.ones([2, 2]), tf.ones([2, 2]))  #  [[2., 2.], [2., 2.]]
```


```python
v = tf.Variable(1.0)
with tf.GradientTape() as tape:
    result = add(v, 1.0)
tape.gradient(result, v)
```

함수 안에서도 함수를 사용할 수 있습니다.


```python
@tf.function
def dense_layer(x, w, b):
    return add(tf.matmul(x, w), b)

dense_layer(tf.ones([3, 2]), tf.ones([2, 2]), tf.ones([2]))
```

<a id="tracing"></a>

## 2. 추적(Tracing)과 다형성(Polymorphism)

Python의 동적 타이핑(typing)은 다양한 인수 타입의 함수를 호출할 수 있음을 의미하며, Python은 각각의 상황에서 다른 작업을 수행합니다.

반면 텐서플로우 그래프는 정적인 데이터 타입과 형태(shape) 차원이 필요합니다. `tf.function`은 정확한 그래프를 생성해야 할 때 함수를 추적하여 이러한 차이를 극복합니다. `tf.function`의 대부분의 세부 사항들은 이러한 재추적을 통해 수행됩니다.

다양한 유형의 인수를 가진 함수를 호출하여 어떤 변화가 있는지 확인할 수 있습니다.


```python
# 함수는 다형성을 가집니다.

@tf.function
def double(a):
    print("Tracing with", a)
    return a + a

print(double(tf.constant(1)))
print()
print(double(tf.constant(1.1)))
print()
print(double(tf.constant("a")))
print()

```

추적을 제어하려면 다음과 같은 기술을 사용합니다:

- 새로운 `tf.function`을 생성합니다. 각각의 `tf.function`객체는 추적을 공유하지 않습니다.
- `get_concrete_function` 메서드는 특정 추적을 가져옵니다.
- `tf.function`을 호출할 때 호출 그래프당 한 번만 추적하기 위해 'input_signature'를 지정합니다.


```python
print("콘크리트(concrete) 추적 내용을 가져옵니다")
double_strings = double.get_concrete_function(tf.TensorSpec(shape=None, dtype=tf.string))
print("추적된 함수를 실행합니다")
print(double_strings(tf.constant("a")))
print(double_strings(a=tf.constant("b")))
print("호환되지 않는 데이터 타입에 콘크리트 추적을 하면 에러를 발생시킵니다")
with assert_raises(tf.errors.InvalidArgumentError):
    double_strings(tf.constant(1))
```


```python
@tf.function(input_signature=(tf.TensorSpec(shape=[None], dtype=tf.int32),))
def next_collatz(x):
    print("Tracing with", x)
    return tf.where(x % 2 == 0, x // 2, 3 * x + 1)

print(next_collatz(tf.constant([1, 2])))
# 입력 시그니처(signature)에 1-D 텐서를 지정했으므로 오류가 발생합니다.
with assert_raises(ValueError):
    next_collatz(tf.constant([[1, 2], [3, 4]]))

```

## 3. 언제 추적을 할까?

다형성을 가진 `tf.function`은 추적을 통해 생성된 콘크리트 함수의 캐시를 가집니다. 캐시 키는 사실 함수의 Args 및 Kwargs에서 생성된 키의 튜플입니다. 

* `tf.Tensor` 인수에 대해 생성된 키는 데이터의 차수와 타입의 개수입니다. 
* Python primitive에 대해 생성된 키는 값입니다. 
* 다른 모든 파이썬의 데이터 타입의 경우 키는 `id()`객체를 기반으로 하므로 클래스의 각 인스턴스에 대해 메서드를 개별적으로 추적할 수 있습니다.



## 4. 파이썬 arg 또는 텐서 args?

파이썬의 인수는 하이퍼 파라미터(hyper parameter) 및 그래프 구조를 제어하는 데에도 사용됩니다.(예: `num_layers=10` 또는 `training=True`, `nonlinearity='relu'`) 그래서 파이썬의 인수가 바뀌면, 그 그래프를 다시 추적해야 합니다.

그러나 파이썬 인수가 그래프 구조를 제어하는 데 사용되지 않을 수 있습니다. 이러한 경우 파이썬 값을 변경하면 불필요한 재추적을 발생시킬 수 있습니다. 예를 들어 AutoGraph가 동적으로 해제되는 상황에서 학습한다고 가정해봅시다. 여러번 추적했을 때, 동일한 그래프를 여러번 생성하기 때문에 다소 비효율적입니다.


```python
def train_one_step():
    pass

@tf.function
def train(num_steps):
    print("Tracing with num_steps = {}".format(num_steps))
    for _ in tf.range(num_steps):
        train_one_step()

train(num_steps=10)
train(num_steps=20)

```

여기서 간단한 해결 방법은 생성된 그래프의 모양(shape)에 영향을 미치지 않는 인수를 텐서(Tensor)에 전달하는 것입니다.


```python
train(num_steps=tf.constant(10))
train(num_steps=tf.constant(20))
```

## 5. `tf.function`의 부작용도(Side effect)

일반적으로 Python 부작용도(예: 객체를 출력하거나 변이를 일으킴)는 추적 중에만 발생합니다. 그러면 어떻게 하면 `tf.function`에서 부작용도를 안정적으로 발생시킬 수 있을까요?

일반적으로 추적을 디버깅할 때에만 Python 부작용도를 사용합니다. 반면에 `tf.Variable.assign`나 `tf.print`, `tf.summary`와 같은 텐서플로우 ops는 텐서플로우 런타임으로 코드를 추적하고 실행할 수 있도록 하는 최선의 방법입니다. 일반적으로 함수형(functional) 스타일을 사용하면 최상의 결과를 얻을 수 있습니다.


```python
@tf.function
def f(x):
    print("Traced with", x)
    tf.print("Executed with", x)

f(1)
f(1)
f(2)

```

`tf.function`을 호출할 때마다 파이썬 코드를 실행하려면 `tf.py_function`을 사용합니다. `tf.function`의 단점은 이동이 가능하거나 특별히 성능이 뛰어나지 않고 분산(Multi-GPU, TPU) 설정에서도 잘 작동하지 않는다는 점입니다. 또한, `tf.py_function`은 미분가능성을 위해 그래프에 연결되어야 하므로 모든 입력/출력을 텐서에 전달합니다.


```python
external_list = []

def side_effect(x):
    print('Python side effect')
    external_list.append(x)

@tf.function
def f(x):
    tf.py_function(side_effect, inp=[x], Tout=[])

f(1)
f(1)
f(1)
assert len(external_list) == 3
# py_function은 1을 tf.constant(1)로 전달하므로 .numpy() 호출이 필요합니다.
assert external_list[0].numpy() == 1

```

## 6. 파이썬 상태(state)를 주의하기

생성자(generator)나 반복자(iterator)와 같은 대부분의 파이썬 기능은 상태를 추적하기 위해 파이썬 런타임을 사용합니다. 일반적 이러한 구조는 즉시 실행 모드에서 예상대로 작동하지만 추적으로 인해 `tf.function` 내에서 예기치 않은 많은 일이 발생할 수 있습니다.

예를 들어, 반복자를 실행하는 것은 파이썬 부작용이므로 추적하는 동안에만 발생합니다.


```python
external_var = tf.Variable(0)
@tf.function
def buggy_consume_next(iterator):
    external_var.assign_add(next(iterator))
    tf.print("external_var의 값:", external_var)

iterator = iter([0, 1, 2, 3])
buggy_consume_next(iterator)
# 다음 값을 사용하지 않고 반복자에서 첫 번째 값이 재사용됩니다.
buggy_consume_next(iterator)
buggy_consume_next(iterator)

```

'tf_function의 내에서 반복자가 생성되고 완전히 사용되면 바르게 작동해야 합니다. 하지만, 전체 반복자는 추적되고 있을 것이고, 이는 거대한 그래프를 만듭니다. 그러나 파이썬 리스트로 된 대규모 인 메모리(in-memory) 데이터셋을 학습하는 경우. 매우 큰 그래프를 생성하기 때문에 `tf.function`은 속도를 높일 수 없을 것입니다.

파이썬 데이터를 반복하려면 tf.data.Dataset으로 감싸 `for x in y` 관용어를 사용하는 것이 가장 안전합니다. AutoGraph는 특별히 `y`가 텐서 또는 tf.data_dataset일 때 안전하게 `for` 루프를 변환할 수 있도록 지원합니다.



```python
def measure_graph_size(f, *args):
    g = f.get_concrete_function(*args).graph
    print("{}({}) contains {} nodes in its graph".format(
          f.__name__, ', '.join(map(str, args)), len(g.as_graph_def().node)))

@tf.function
def train(dataset):
    loss = tf.constant(0)
    for x, y in dataset:
        loss += tf.abs(y - x) # Some dummy computation.
    return loss

small_data = [(1, 1)] * 2
big_data = [(1, 1)] * 10
measure_graph_size(train, small_data)
measure_graph_size(train, big_data)

measure_graph_size(train, tf.data.Dataset.from_generator(
    lambda: small_data, (tf.int32, tf.int32)))
measure_graph_size(train, tf.data.Dataset.from_generator(
    lambda: big_data, (tf.int32, tf.int32)))
```


파이썬/넘파이 데이터를 데이터셋에 감쌀 때는 `tf.data.Dataset.from_generator`나 `tf.data.Dataset.from_tensors`를 사용합니다. 전자는 파이썬에 데이터를 보관하고 성능의 영향을 미칠 수 있는 `tf.py_function`을 통해 데이터를 가져오게 되며, 후자는 그래프에 큰 `tf.constant()` 노드로 묶어서 메모리에 영향을 미칠 수 있습니다.

TFRecordDataset/CsvDataset/etc를 통해 파일에서 데이터를 읽는 것이 가장 효과적인 방법인데, 이는 텐서플로우 자체가 파이썬을 수반하지 않고도 데이터의 비동기적 로딩(asynchronous loading)과 프리페치(prefetching)를 관리할 수 있기 때문입니다.

## 7. 자동 제어 종속변수

일반적인 데이터 흐름(dataflow) 그래프에서 프로그래밍 모델과 같은 함수의 매력적인 특성은 함수가 코드의 의도된 동작에 대한 정보를 런타임에 더 많이 제공할 수 있다는 것입니다.

예를 들어, 동일한 변수에 대한 읽기 및 쓰기가 여러 개 있는 코드를 작성할 때 데이터 흐름 그래프가 원래 의도한 작업 순서를 자연스럽게 인코딩하지 않을 수 있습니다. `tf.function`에서는 파이썬 코드의 문장 실행 순서를 참조하여 실행 순서의 모호성을 해결합니다. 이런 방법으로, `tf.function`에서 네트워크 연결 상태를 추적할 수 있는 작업을 정렬하면 즉시 실행 모드의 의미를 복제합니다.

즉, 수동 제어 종속 변수를 추가할 필요가 없으며, `tf.function`은 코드가 올바르게 실행되기 위해 필요한 최소한의 제어 종속변수를 추가하는 것으로도 가능할 정도로 충분히 스마트합니다.


```python
# 자동 제어 종속 변수

a = tf.Variable(1.0)
b = tf.Variable(2.0)

@tf.function
def f(x, y):
    a.assign(y * b)
    b.assign_add(x * a)
    return a + b

f(1.0, 2.0)  # 10.0

```

## 8. 변수

`tf.function`에서는 코드의 의도된 실행 순서를 활용하여 변수 생성과 활용을 매우 쉽게 할 수 있습니다. 그러나 매우 중요한 사항이 하나 있습니다. 변수를 사용하면 즉시 실행 모드와 그래프 모드에서 다르게 동작하는 코드를 작성하게 될 수도 있다는 것입니다.

특히, 각 호출마다 새 변수를 생성할 때 이 문제가 발생합니다. 추적 의미(semantics) 때문에 `tf.function`은 각 호출마다 동일한 변수를 재사용하지만, 즉시 실행 모드는 각 호출마다 새로운 변수를 생성합니다. 이러한 실수를 방지하기 위해 'tf.function'은 위험한 변수 생성 동작을 감지할 경우 오류를 발생시킵니다.


```python
@tf.function
def f(x):
    v = tf.Variable(1.0)
    v.assign_add(x)
    return v

with assert_raises(ValueError):
    f(1.0)
```

하지만 모호하지 않은 코드는 괜찮습니다.




```python
v = tf.Variable(1.0)

@tf.function
def f(x):
    return v.assign_add(x)

print(f(1.0))  # 2.0
print(f(2.0))  # 4.0

```

또한 함수가 처음 실행될 때만 해당 변수를 생성함이 증명되면 tf_function 내부에 변수를 생성할 수도 있습니다.


```python
class C:
    pass

obj = C()
obj.v = None

@tf.function
def g(x):
    if obj.v is None:
        obj.v = tf.Variable(1.0)
    return obj.v.assign_add(x)

print(g(1.0))  # 2.0
print(g(2.0))  # 4.0
```

변수 초기화는 함수 인수 및 다른 변수의 값에 따라 달라질 수 있습니다. 제어 종속성을 생성하기 위해 사용하는 것과 동일한 방법으로 올바른 초기화 순서를 알아낼 수 있습니다.


```python
state = []
@tf.function
def fn(x):
    if not state:
        state.append(tf.Variable(2.0 * x))
        state.append(tf.Variable(state[0] * 3.0))
    return state[0] * x * state[1]

print(fn(tf.constant(1.0)))
print(fn(tf.constant(3.0)))

```

## 9. 오토그래프(AutoGraph) 사용하기

[오토그래프](https://www.tensorflow.org/guide/function) 라이브러리는 tf_function과 완벽하게 통합되어 있으며, 그래프에서 동적으로 실행되도록 텐서에 의존하는 조건 및 루프를 다시 작성합니다.

`tf_cond`와 `tf.while_loop`는 계속 `tf.function`으로 작동하지만, 제어 흐름(control flow)이 있는 코드는 명령어로 작성하면 쓰고, 이해하기가 더 쉬운 경우가 많습니다.


```python
# Simple loop

@tf.function
def f(x):
    while tf.reduce_sum(x) > 1:
        tf.print(x)
        x = tf.tanh(x)
    return x

f(tf.random.uniform([5]))
```

궁금하면 오토그래프를 생성하는 코드를 확인해볼 수 있습니다.


```python
def f(x):
    while tf.reduce_sum(x) > 1:
        tf.print(x)
        x = tf.tanh(x)
    return x

print(tf.autograph.to_code(f))
```

## 10. 오토그래프: 조건문

오토그래프는 `if` 문을 동등한 `tf.cond` 호출로 변환합니다.

이 교체는 조건이 텐서인 경우 이루어집니다. 그렇지 않으면 추적 중에 조건이 실행됩니다.

다음은 결과 그래프가 `tf.cond`를 사용하는지 확인하는 함수입니다.


```python
def test_tf_cond(f, *args):
    g = f.get_concrete_function(*args).graph
    if any(node.name == 'cond' for node in g.as_graph_def().node):
        print("{}({}) uses tf.cond.".format(
            f.__name__, ', '.join(map(str, args))))
    else:
        print("{}({}) executes normally.".format(
            f.__name__, ', '.join(map(str, args))))

    print("  result: ",f(*args).numpy())
```

이 교체는 조건이 텐서인 경우 이루어집니다. 그렇지 않으면 추적 중에 조건이 실행됩니다.

파이썬의 `True`를 전달하면 조건부가 정상적으로 실행됩니다.


```python
@tf.function
def dropout(x, training=True):
    if training:
        x = tf.nn.dropout(x, rate=0.5)
    return x
```


```python
test_tf_cond(dropout, tf.ones([10], dtype=tf.float32), True)
```

하지만 텐서를 전달하면 파이썬의 `if`문은 `tf.cond`로 교체됩니다.


```python
test_tf_cond(dropout, tf.ones([10], dtype=tf.float32), tf.constant(True))
```

tf.second에는 여러 가지 미묘한 점이 있습니다.

조건의 양쪽을 모두 추적한 다음 조건에 따라 런타임에 적절한 분기점(branch)을 선택하는 방식으로 작동합니다. 양쪽을 추적하면 파이썬 코드가 예기치 않게 실행될 수 있습니다.


```python
@tf.function
def f(x):
    if x > 0:
        x = x + 1.
        print("Tracing `then` branch")
    else:
        x = x - 1.
        print("Tracing `else` branch")
    return x
```


```python
f(-1.0).numpy()
```


```python
f(1.0).numpy()
```


```python
f(tf.constant(1.0)).numpy()
```

한 분기점이 다운스트림에 사용된 텐서를 생성하는 경우 다른 분기점도 해당 텐서를 생성해야 합니다.


```python
@tf.function
def f():
    if tf.constant(True):
        x = tf.ones([3, 3])
    return x

# 모든 분기점은 `x`를 정의해야하므로 에러를 발생시킵니다.
with assert_raises(ValueError):
    f()
```

특정 제어 흐름 섹션이 오토그래프에 의해 변환되지 않도록 하려면 개체를 명시적으로 파이썬 유형으로 변환하여 오류가 발생하도록 합니다.


```python
@tf.function
def f(x, y):
    if bool(x):
        y = y + 1.
        print("Tracing `then` branch")
    else:
        y = y - 1.
        print("Tracing `else` branch")
    return y
```


```python
f(True, 0).numpy()
```


```python
f(False, 0).numpy()
```


```python
with assert_raises(TypeError):
  f(tf.constant(True), 0.0)
```

## 11. 오토그래프와 반복문

오토그래프는 반복문을 변환시키는 데 몇 가지 간단한 규칙이 있습니다.

- `for`: 반복할 수 있는 경우 텐서로 변환합니다.
- `while`: while 조건문이 텐서에 따라 달라지는 경우에 변환합니다.

반복문이 변환되면 `tf.while_loop`이나 특별한 경우 `for x in tf.data.Dataset`으로 동적으로 역할을 해제(unroll)하고 `tf.data.Dataset.reduce`로 변환합니다.

반복문이 변환되지 않으면, 정적으로 해제됩니다.


```python
def test_dynamically_unrolled(f, *args):
    g = f.get_concrete_function(*args).graph
    if any(node.name == 'while' for node in g.as_graph_def().node):
        print("{}({}) uses tf.while_loop.".format(
            f.__name__, ', '.join(map(str, args))))
    elif any(node.name == 'ReduceDataset' for node in g.as_graph_def().node):
        print("{}({}) uses tf.data.Dataset.reduce.".format(
            f.__name__, ', '.join(map(str, args))))
    else:
        print("{}({}) gets unrolled.".format(
            f.__name__, ', '.join(map(str, args))))
```

### 11.1 For문

아래는 정적 역할 해제를 하는 `tf.function`입니다:


```python
@tf.function
def for_in_range():
    x = 0
    for i in range(5):
        x += i
    return x

test_dynamically_unrolled(for_in_range)
```


```python
@tf.function
def for_in_tfrange():
    x = tf.constant(0, dtype=tf.int32)
    for i in tf.range(5):
        x += i
    return x

test_dynamically_unrolled(for_in_tfrange)
```


```python
@tf.function
def for_in_tfdataset():
    x = tf.constant(0, dtype=tf.int64)
    for i in tf.data.Dataset.range(5):
        x += i
    return x

test_dynamically_unrolled(for_in_tfdataset)
```


```python
@tf.function
def while_py_cond():
    x = 5
    while x > 0:
        x -= 1
    return x

test_dynamically_unrolled(while_py_cond)
```


```python
@tf.function
def while_tf_cond():
    x = tf.constant(5)
    while x > 0:
        x -= 1
    return x

test_dynamically_unrolled(while_tf_cond)
```

텐서에 의존하는 `break`나 `return` 조항이 있다면 최상위 조건이나 반복될 수 있는 것도 텐서여야 합니다.

다음 예를 비교합니다:


```python
@tf.function
def while_py_true_py_break(x):
    while True:  # py true
        if x == 0: # py break
            break
        x -= 1
    return x

test_dynamically_unrolled(while_py_true_py_break, 5)
```


```python
@tf.function
def buggy_while_py_true_tf_break(x):
    while True:   # py true
        if tf.equal(x, 0): # tf break
            break
        x -= 1
    return x

with assert_raises(TypeError):
    test_dynamically_unrolled(buggy_while_py_true_tf_break, 5)
```


```python
@tf.function
def while_tf_true_tf_break(x):
    while tf.constant(True): # tf true
        if x == 0:  # py break
            break
        x -= 1
    return x

test_dynamically_unrolled(while_tf_true_tf_break, 5)
```


```python
@tf.function
def buggy_py_for_tf_break():
    x = 0
    for i in range(5):  # py for
        if tf.equal(i, 3): # tf break
            break
        x += i
    return x

with assert_raises(TypeError):
    test_dynamically_unrolled(buggy_py_for_tf_break)
```


```python
@tf.function
def tf_for_py_break():
    x = 0
    for i in tf.range(5): # tf for
        if i == 3:  # py break
            break
        x += i
    return x

test_dynamically_unrolled(tf_for_py_break)
```

동적으로 해제되지 않는 반복문에서 결과를 누적하려면 `tf.TensorArray`를 사용합니다.


```python
batch_size = 2
seq_len = 3
feature_size = 4

def rnn_step(inp, state):
    return inp + state

@tf.function
def dynamic_rnn(rnn_step, input_data, initial_state):
    # [batch, time, features] -> [time, batch, features]
    input_data = tf.transpose(input_data, [1, 0, 2])
    max_seq_len = input_data.shape[0]

    states = tf.TensorArray(tf.float32, size=max_seq_len)
    state = initial_state
    for i in tf.range(max_seq_len):
        state = rnn_step(input_data[i], state)
        states = states.write(i, state)
    return tf.transpose(states.stack(), [1, 0, 2])
  
dynamic_rnn(rnn_step,
            tf.random.uniform([batch_size, seq_len, feature_size]),
            tf.zeros([batch_size, feature_size]))
```

### 11.2 Gotcha's

`tf.cond`와 마찬가지로 `tf.while_loop`에도 여러 가지 미묘한 점이 있습니다.


#### - 제로 반복자(Zero iterations)

반복문은 0회 실행될 수 있으므로, while_loop의 다운스트림에 사용되는 모든 텐서를 반복문 위에서 초기화해야 합니다.

다음은 잘못된 코드의 예입니다.


```python
@tf.function
def buggy_loop_var_uninitialized():
    for i in tf.range(3):
        x = i
    return x

with assert_raises(ValueError):
    buggy_loop_var_uninitialized()
```

그리고 아래는 맞는 코드입니다:


```python
@tf.function
def f():
    x = tf.constant(0)
    for i in tf.range(3):
        x = i
    return x

f()
```

#### - 일관성 있는 형태와 타입

모든 반복문 변수의 형태/데이터 타입이 각 반복과 일치해야 합니다.

다음은 텐서의 유형을 변경하려는 잘못된 예입니다.


```python
@tf.function
def buggy_loop_type_changes():
    x = tf.constant(0, dtype=tf.float32)
    for i in tf.range(3): # tf.int32타입의 텐서를 생성합니다.
        x = i
    return x

with assert_raises(TypeError):
    buggy_loop_type_changes()
```

다음은 반복하면서 텐서의 형태를 변환하려고하는 잘못된 예시입니다:


```python
@tf.function
def buggy_concat():
    x = tf.ones([0, 10])
    for i in tf.range(5):
        x = tf.concat([x, tf.ones([1, 10])], axis=0)
    return x

with assert_raises(ValueError):
    buggy_concat()
```


```python
@tf.function
def concat_with_padding():
    x = tf.zeros([5, 10])
    for i in tf.range(5):
        x = tf.concat([x[:i], tf.ones([1, 10]), tf.zeros([4-i, 10])], axis=0)
        x.set_shape([5, 10])
    return x

concat_with_padding()

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
