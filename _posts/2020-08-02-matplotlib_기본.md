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
