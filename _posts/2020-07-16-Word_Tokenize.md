---
header:
  teaser: /assets/images/ai/로지스틱시그모이드.PNG
title:  "Word_Tokenize"
excerpt: "Word_Tokenize, Word_Cloud"
toc: true
toc_sticky: true
categories:
  - AI_개념
tags:
  - AI
last_modified_at: 2020-07-16
---
```python
!pip install nltk 
!pip install konlpy 
!pip install kss 
```
```python
import nltk

nltk.download('punkt')
```

# 다양한 문장 처리 tokenize

```python
from nltk.tokenize import word_tokenize

text = """
I'm only one call away.
I'll be there to save the day.
Superman got nothing on me.
I'm only one call away.
Call me, baby, if you need a friend.
I just wanna give you love.
Come on, come on, come on.
"""

print(word_tokenize(text))
```
```python
from nltk.tokenize import WordPunctTokenizer

print(WordPunctTokenizer().tokenize(text))

#(') 포함
```
```python
from nltk.tokenize import TreebankWordTokenizer

print(TreebankWordTokenizer().tokenize(text))

#('.) 포함 단어 구분
```
```python
from nltk.tokenize import RegexpTokenizer

print(RegexpTokenizer('\w+').tokenize(text))

# 정규 표현식으로 구분
```
```python
from nltk.tokenize import sent_tokenize

print(sent_tokenize(text))

#(.)을 사용해서 문장 구분
```
```python
from konlpy.tag import Okt, Kkma
import kss
# 한글에도 사용할 수 있다
ktext = '안녕하세요? 저는 임준형이라고 합니다'

print(Okt().morphs(ktext))
```
```python
print(Kkma().morphs(ktext))
```
```python
from nltk.stem import PorterStemmer

words = word_tokenize(text)
for word in words:
    print(PorterStemmer().stem(word), end = ' ')
```
```python
nltk.download('stopwords')
# stopwords
```
```python
from nltk.corpus import stopwords

sw = stopwords.words('english')
print(sw)

# 문장에서 중요하지 않는 단어를 정리해서 담아둔다
```
```python
print(words)
```

# 중요하지 않는 단어 제거하기

```python
# 내 버전
str = []
for i in range(len(words)):
    check  = True
    for j in range(len(sw)):
        if words[i].upper() == sw[j].upper():
            check = False
            break
    if check:
        str.append(words[i])
print(str)
```
```python
# 선생님 버전
sw = [',','.']
# ,와 .만 제거
sw_removed = []
for i in words:
    if i.lower() not in sw:
        sw_removed.append(i)
        
print(sw_removed)
```

# 단어를 빈도수를 기준으로 정렬

```python
from collections import Counter

count_list = Counter(sw_removed)
print(count_list)
```
```python
# 가장 많이 나온 단어 10개 출력
common_cl = count_list.most_common(10)
print(common_cl)
```

# 우선순위 주기

```python
# 내 버전
common_cl_dict = {}

for i in range(10):
    common_cl_dict[common_cl[i][0]] = i

print(common_cl_dict)
```
```python
# 선생님 버전

common_cl_dict = {}
i = 0
for (key, value) in common_cl:
    common_cl_dict[key] = i
    i = i+1

print(common_cl_dict)
```
# One-Hot Vector

```python
# 선생님 버전
oh_vector_list = []

for value in common_cl_dict.values():
    oh_vector = [0] * len(common_cl_dict)
    oh_vector[value] = 1
    oh_vector_list.append(oh_vector)

print(oh_vector_list)
```
# 워드 클라우드
```python
from wordcloud import WordCloud
import matplotlib.pyplot as plt

word_wc = WordCloud(background_color = 'white')
plt.imshow(word_wc.generate_from_frequencies(count_list))
plt.show()
```

![Word_Cloud](/assets/images/ai/Word_Cloud.PNG)  
