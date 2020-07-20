---
header:
  teaser: /assets/images/ai/Word_Cloud_Alice.PNG
title:  "WordCloud"
excerpt: "WordCloud"
categories:
  - AI_개념
tags:
  - AI
  - AI_Algorithm
  - WordCloud
  - 자연어 처리
  - python
last_modified_at: 2020-07-20
---

![Word_Cloud_Alice.PNG](/assets/images/ai/Word_Cloud_Alice.PNG) 

# import
```python
from wordcloud import WordCloud
```

# WordCloud

```python
my_wc = WordCloud(max_font_size = 70,font_path = "c:/Windows/Fonts/malgun.ttf", background_color = 'white',max_words = 2000, mask = alice_mask)
```
* max_font_size 

      사이즈의 최대값을 정해준다
  
* font_path

      fontName이 있는 경로를 정해준다
      
* background_color

      배경색을 정해준다.
      
# matplotlib
```python
plt.figure(figsize=(15,15))
plt.imshow(my_wc.generate_from_frequencies(count_list),interpolation='bilinear')
plt.axis("off")
plt.show()
```
* figsize 

      WordCloud의 크기를 정해준다
* interpolation = 'bilinear'

      WordCloud를 더 부드럽게 출력되게 해준다
