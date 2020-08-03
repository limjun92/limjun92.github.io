# 아스키 코드

* ord() : 문자의 아스키 코드 값을 가져온다
* chr() : 아스키 코드 값으로 문자를 가져온다

```python
ord('A') # 65
chr(66)  # B
```

# isnumeric()
* a 문자가 숫자이면 True 숫자가 아니면 False

```python
a = '1'
a.isnumeric() # True
b = 'a'
b.isnumeric() # False
```

# lambda 
* 함수를 만드는 또 다른 방식
* 함수의 이름이 없이 사용 가능

```python
def square(x):
  return x * x
  
# 위 아래는 동일함
  
square = lambda x: x * x
```

```python
def _first_letter(string):
    return string[0] if string else ''

# 위 아래는 동일함

first_letter = lambda string: string[0] if string else ''
```

# sorted

```python
def get_eng_title(row):
  split = row.split(',')
  return split[1]

# 위 아래는 동일함

# lambda를 사용해보자
get_eng_title = lambda row: row:split(',')[1]

 
sorted(movies, key = get_eng_title)
# or
sorted(movies, key = lambda row: row.split(',')[1])

```

# assert()

* () 안의 값이 True이면 그냥 지나감
* () 안의 값이 False이면 정지함

```
def square1(x):
  return x * x

square2 = lambda x: x * x

# 두 값이 같으면 통과, 아니면 에러
assert(square1(3) == square2(3))
```

# 함수를 리턴하는 함수

```python
def adder(n):
    def helper(x):
        return x + n
    return helper

add_three = adder(3)
print(add_three(6)) # 9
```
