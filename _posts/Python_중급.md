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

# map()

* map(함수, 리스트)
* 리스트 안의 값들을 각각 함수를 적용해서 리스트로  준다 
* map은 리스트를 만들지 않고 map이라는 타입을 가진다
* map의 내용을 사용할때 해당함수를 적용시켜서 보여준다  
  (효율적이다. 리스트보다 빠르다)
* 결과물을 확인할때 list(type가 map인 데이터)
```python
def get_eng_title(row):
  split = row.split(',')
  return split[1]
  
eng_tities = [get_eng_title(row) for row in movies]

# 위와 아래는 같다

eng_titles = map(get_eng_title, movies)

# 위와 아래는 같다
```

* lambda를 사용해 함수를 단순화
* 위의 코드와 결과가 같다

```python
eng_titles = map(lambda row: row.split(',')[1], movies)
```

```python 
def get_eng_title(row):
  split = row.split(',')
  return split[1]

1.
[get_eng_title(row) for row in movies]
2.
map(get_eng_title, movies)
3.
[row.split(',')[1] for row in movies]
4.
map(lambda row: row.split(',')[1], movies)

# 위의 4가지 코드 모두 같은 코드이다
```

## 예제

    books.csv 파일을 읽어서 
    책들의 제목을 리스트로 리턴하는 함수 

```python
# CSV 모듈을 임포트합니다.
import csv

def get_titles(books_csv):
    with open(books_csv) as books:
        reader = csv.reader(books, delimiter=',')
        get_title = lambda row: row[0]
        # get_title이라는 함수를 만들어준다
        # row[0]번째를 가져온다
        
        titles = map(get_title,reader)
        # 모든 reader데이터에 get_title을 적용시켜준다
        
        return list(titles)
        
books = 'books.csv'
titles = get_titles(books)
for title in titles:
     print(title)
```

# filter()

* filter(함수, 리스트)
* 리스트를 함수에 적용한후 True인 값을 가지는 리스트만 보여준다
* map과 비슷하게 filter 타입을 가진다
* 연산을 뒤로 미룬다

```python
def starts_with_r(word):
  return word.startswith('r')
 

r_words = filter(starts_with_r, words)

# 위와 아래는 같다

starts_with_r = lambda w: w.startswith('r')
words = ['real', 'man', 'rhythm',...]
r_words = filter(starts_with_r, words) 
```

## 예제

    books.csv 파일을 읽어서 페이지 수가 250이 넘는 
    책들의 제목을 리스트로 리턴하는 
    get_titles_of_long_books() 함수를 완성

```python
# CSV 모듈을 임포트합니다.
import csv

def get_titles_of_long_books(books_csv):
    with open(books_csv) as books:
        reader = csv.reader(books, delimiter=',')
        
        is_long = lambda row: int(row[3])>250
        get_title = lambda row: row[0]
        
        long_books = filter(is_long, reader)
        long_book_titles = map(get_title, long_books)
        
        return list(long_book_titles)

books  = 'books.csv'
titles = get_titles_of_long_books(books)
for title in titles:
    print(title)
```
