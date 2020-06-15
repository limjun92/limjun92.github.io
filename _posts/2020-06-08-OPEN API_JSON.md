---
title:  "OPEN API JSON 읽어오기"
excerpt: "파이썬 사용 공공데이터에서 코로나 마스크 정보 읽어오기"
toc: true
toc_sticky: true
categories:
  - AI_개념
tags:
  - JSON
  - AI
  - OPEN API
last_modified_at: 2020-06-08
---
#### [코로나19_mask_예제](https://github.com/limjun92/limjun92.github.io/blob/master/ipynb/corona19_mask.ipynb)

### JSON(javascript Object Notation)
    데이터를 효율적으로 저장하고 교환하는데 사용
    텍스트 데이터 형식
    이름 & 값
    정렬된 값의 리스트
    python(list, tuple) => JSON(array)
    JSON(array) => Python(list)

* 딕셔너리
  * 키(key) 값(value)

* json형식의 데이터를 열어 파이썬 객체로 읽어와 주는 것
```python
import json 
with open('student_file.json) as json_file:
  json_data = json.load(json_file)
json_data
```
```python
st_json = json.dumps(student_data, indent-4, sort_keys = True)
st_json
```
* indent-4 는 들여쓰기를 해주는 설정이다
* sort_keys = True 는키를 기준으로 정렬해준다.

### OPNE API

    * 인터넷 이용자가 웹 검색 경과 및 사용자 인터페이스 등을 수동적으로
    제공받는데 그치지 않고 직접 응용 프로그램과 서비스를 개발할 수 있고록
    공개된 API
    * 인터넷 이용자가 웹 검색 결과 즉 데이터를 우리가 지정한 조건에 맞게끔
    가져올 수 있는 도구



```python
import requests

url = "http://www.xxxx.xx.xx/xxx...xx.json?key=xxx"
res = requests.get(url)
text = res.text
MD_json = json.loads(text) 

print(json.dumps(MD_json,indent=4,sort_keys=True))
print(MD_json.key())
print(MD_json['boxOfficeResult'].keys())
for i in MD_json['box']['xxx']:
  print(i['rank'],i[xx]...)
```
* pandas 형식으로 저장하고 싶을때
```python
import pandas as pd
movie = []
for i in MD_json['box']['xxx']:
  movie.append(i['rank'],i[xx]...)
data = pd.DataFrame(movie)
```
