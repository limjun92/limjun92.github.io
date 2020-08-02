---
title:  "matplotlib_기본"
excerpt: ""
toc: true
toc_sticky: true
categories:
  - Python
tags:

last_modified_at: 2020-08-02
---

* Mathematical Plot Library
* 파이썬에서 그래프를 그릴 수 있게 하는 라이브러리
* 꺾은선 그래프, 막대 그래프 등을 모두 지원

```python 
import matplotlib.pyplot as plt

years = [2013, 2014, 2015, 2016, 2017]
temperatures = [5, 10, 15, 20, 17]

def draw_graph():
    pos = range(len(years))  # [0, 1, 2, 3, 4]
    
    # 막대 그래프를 그린다
    # 각 막대를 가운대 정렬
    plt.bar(pos, temperatures, align='center')
    
    # pos를 years로 지정
    plt.xticks(pos, years)
    plt.show()
    
draw_graph()
```

# matplotlib으로 그래프 설정
* 제목달기
* 축 별로 라벨 추가하기
* 막대 그래프의 tick 그리기
* 여백 조정하기

```python
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 날짜 별 온도 데이터를 세팅합니다.
dates = ["1월 {}일".format(day) for day in range(1, 32)]
temperatures = list(range(1, 32))

# 막대 그래프의 막대 위치를 결정하는 pos를 선언합니다.
pos = range(len(dates))

# 한국어를 보기 좋게 표시할 수 있도록 폰트를 설정합니다.
font = fm.FontProperties(fname='./NanumBarunGothic.ttf')

# 막대의 높이가 빈도의 값이 되도록 설정합니다.
plt.bar(pos, temperatures, align='center')

# 각 막대에 해당되는 단어를 입력합니다.
plt.xticks(pos, dates, rotation='vertical', fontproperties=font)

# 그래프의 제목을 설정합니다.
plt.title('1월 중 기온 변화', fontproperties=font)

# Y축에 설명을 추가합니다.
plt.ylabel('온도', fontproperties=font)

# 단어가 잘리지 않도록 여백을 조정합니다.
plt.tight_layout()

# 그래프를 표시합니다.
plt.show()
```
