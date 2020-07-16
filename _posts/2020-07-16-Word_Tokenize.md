---
header:
  teaser: /assets/images/ai/Word_Cloud.PNG
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
['I', "'m", 'only', 'one', 'call', 'away', '.', 'I', "'ll", 'be', 'there', 'to', 'save', 'the', 'day', '.', 'Superman', 'got', 'nothing', 'on', 'me', '.', 'I', "'m", 'only', 'one', 'call', 'away', '.', 'Call', 'me', ',', 'baby', ',', 'if', 'you', 'need', 'a', 'friend', '.', 'I', 'just', 'wan', 'na', 'give', 'you', 'love', '.', 'Come', 'on', ',', 'come', 'on', ',', 'come', 'on', '.']
```python
from nltk.tokenize import WordPunctTokenizer

print(WordPunctTokenizer().tokenize(text))

#(') 포함
```
['I', "'", 'm', 'only', 'one', 'call', 'away', '.', 'I', "'", 'll', 'be', 'there', 'to', 'save', 'the', 'day', '.', 'Superman', 'got', 'nothing', 'on', 'me', '.', 'I', "'", 'm', 'only', 'one', 'call', 'away', '.', 'Call', 'me', ',', 'baby', ',', 'if', 'you', 'need', 'a', 'friend', '.', 'I', 'just', 'wanna', 'give', 'you', 'love', '.', 'Come', 'on', ',', 'come', 'on', ',', 'come', 'on', '.']
```python
from nltk.tokenize import TreebankWordTokenizer

print(TreebankWordTokenizer().tokenize(text))

#('.) 포함 단어 구분
```
['I', "'m", 'only', 'one', 'call', 'away.', 'I', "'ll", 'be', 'there', 'to', 'save', 'the', 'day.', 'Superman', 'got', 'nothing', 'on', 'me.', 'I', "'m", 'only', 'one', 'call', 'away.', 'Call', 'me', ',', 'baby', ',', 'if', 'you', 'need', 'a', 'friend.', 'I', 'just', 'wan', 'na', 'give', 'you', 'love.', 'Come', 'on', ',', 'come', 'on', ',', 'come', 'on', '.']
```python
from nltk.tokenize import RegexpTokenizer

print(RegexpTokenizer('\w+').tokenize(text))

# 정규 표현식으로 구분
```
['I', 'm', 'only', 'one', 'call', 'away', 'I', 'll', 'be', 'there', 'to', 'save', 'the', 'day', 'Superman', 'got', 'nothing', 'on', 'me', 'I', 'm', 'only', 'one', 'call', 'away', 'Call', 'me', 'baby', 'if', 'you', 'need', 'a', 'friend', 'I', 'just', 'wanna', 'give', 'you', 'love', 'Come', 'on', 'come', 'on', 'come', 'on']
```python
from nltk.tokenize import sent_tokenize

print(sent_tokenize(text))

#(.)을 사용해서 문장 구분
```
["\nI'm only one call away.", "I'll be there to save the day.", 'Superman got nothing on me.', "I'm only one call away.", 'Call me, baby, if you need a friend.', 'I just wanna give you love.', 'Come on, come on, come on.']
```python
from konlpy.tag import Okt, Kkma
import kss
# 한글에도 사용할 수 있다
ktext = '안녕하세요? 저는 임준형이라고 합니다'

print(Okt().morphs(ktext))
```
['안녕하세요', '?', '저', '는', '임준', '형', '이라고', '합니다']
```python
print(Kkma().morphs(ktext))
```
['안녕', '하', '세요', '?', '저', '는', '임', '준형', '이', '라고', '하', 'ㅂ니다']
```python
from nltk.stem import PorterStemmer

words = word_tokenize(text)
for word in words:
    print(PorterStemmer().stem(word), end = ' ')
```
I 'm onli one call away . I 'll be there to save the day . superman got noth on me . I 'm onli one call away . call me , babi , if you need a friend . I just wan na give you love . come on , come on , come on . 
```python
nltk.download('stopwords')
# stopwords
```
True
```python
from nltk.corpus import stopwords

sw = stopwords.words('english')
print(sw)

# 문장에서 중요하지 않는 단어를 정리해서 담아둔다
```
['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
```python
print(words)
```
['I', "'m", 'only', 'one', 'call', 'away', '.', 'I', "'ll", 'be', 'there', 'to', 'save', 'the', 'day', '.', 'Superman', 'got', 'nothing', 'on', 'me', '.', 'I', "'m", 'only', 'one', 'call', 'away', '.', 'Call', 'me', ',', 'baby', ',', 'if', 'you', 'need', 'a', 'friend', '.', 'I', 'just', 'wan', 'na', 'give', 'you', 'love', '.', 'Come', 'on', ',', 'come', 'on', ',', 'come', 'on', '.']

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
["'m", 'one', 'call', 'away', '.', "'ll", 'save', 'day', '.', 'Superman', 'got', 'nothing', '.', "'m", 'one', 'call', 'away', '.', 'Call', ',', 'baby', ',', 'need', 'friend', '.', 'wan', 'na', 'give', 'love', '.', 'Come', ',', 'come', ',', 'come', '.']
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
['I', "'m", 'only', 'one', 'call', 'away', 'I', "'ll", 'be', 'there', 'to', 'save', 'the', 'day', 'Superman', 'got', 'nothing', 'on', 'me', 'I', "'m", 'only', 'one', 'call', 'away', 'Call', 'me', 'baby', 'if', 'you', 'need', 'a', 'friend', 'I', 'just', 'wan', 'na', 'give', 'you', 'love', 'Come', 'on', 'come', 'on', 'come', 'on']

# 단어를 빈도수를 기준으로 정렬

```python
from collections import Counter

count_list = Counter(sw_removed)
print(count_list)
```
Counter({'I': 4, 'on': 4, "'m": 2, 'only': 2, 'one': 2, 'call': 2, 'away': 2, 'me': 2, 'you': 2, 'come': 2, "'ll": 1, 'be': 1, 'there': 1, 'to': 1, 'save': 1, 'the': 1, 'day': 1, 'Superman': 1, 'got': 1, 'nothing': 1, 'Call': 1, 'baby': 1, 'if': 1, 'need': 1, 'a': 1, 'friend': 1, 'just': 1, 'wan': 1, 'na': 1, 'give': 1, 'love': 1, 'Come': 1})
```python
# 가장 많이 나온 단어 10개 출력
common_cl = count_list.most_common(10)
print(common_cl)
```
[('I', 4), ('on', 4), ("'m", 2), ('only', 2), ('one', 2), ('call', 2), ('away', 2), ('me', 2), ('you', 2), ('come', 2)]

# 우선순위 주기

```python
# 내 버전
common_cl_dict = {}

for i in range(10):
    common_cl_dict[common_cl[i][0]] = i

print(common_cl_dict)
```
{'I': 0, 'on': 1, "'m": 2, 'only': 3, 'one': 4, 'call': 5, 'away': 6, 'me': 7, 'you': 8, 'come': 9}
```python
# 선생님 버전

common_cl_dict = {}
i = 0
for (key, value) in common_cl:
    common_cl_dict[key] = i
    i = i+1

print(common_cl_dict)
```
{'I': 0, 'on': 1, "'m": 2, 'only': 3, 'one': 4, 'call': 5, 'away': 6, 'me': 7, 'you': 8, 'come': 9}
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
[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
# 워드 클라우드
```python
from wordcloud import WordCloud
import matplotlib.pyplot as plt

word_wc = WordCloud(background_color = 'white')
plt.imshow(word_wc.generate_from_frequencies(count_list))
plt.show()
```

![Word_Cloud](/assets/images/ai/Word_Cloud.PNG)  
