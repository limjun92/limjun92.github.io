---
title:  "python with sqlite"
excerpt: "python with sqlite"
toc: true
toc_sticky: true
categories:
  - AI_개념
tags:
  - Python
  - sqlite
last_modified_at: 2020-07-10
---
# python with sqlite기본 
```
import sqlite3
conn = sqlite3.connect('world.db')
# jupyter lab에서는 db가 자동으로 생성된다.
cursor = conn.cursor()
# 커서를 만든다

cursor.execute("create table samsung(date text, open int, high int, low int, closing int, volumn int)")
cursor.execute("insert into samsung values('20.07.08',55800,55900,53400,55000,357)")
cursor.execute("insert into samsung values('20.07.07',55800,55900,53400,55000,357)")
# 다음과 같은 방식으로 sql문을 사용할 수 있다.

for i in cursor.execute("select * from samsung"):
    print(f'{i[0]} {i[1]}')
# 다음을 통해 데이터를 확인 할수 있다.

```

# pandas 형태로 변환
```
import pandas as pd
cursor.execute("SELECT * FROM samsung")
rows = cursor.fetchall()
# 모든 데이터를 가져온다
cols = [column[0] for column in cursor.description]
# 모든 속성이름을 가져온다
data_df = pd.DataFrame.from_records(data=rows, columns=cols)
print(data_df)
```
   date   open   high    low  closing  volumn
0  20.07.08  55800  55900  53400    55000     357
1  20.07.07  55800  55900  53400    55000     357
2  20.07.08  55800  55900  53400    55000     357
3  20.07.07  55800  55900  53400    55000     357
# insert
```
def write(title, content, username, cursor):
    cursor.execute("insert into feed values(?,?,?)",(title,content,username))
```

# like 사용하기
```
def read(title, cursor):
  for i in cursor.execute("select * from feed where title like ?", ('%'+title+'%',)):
          print(f'{i[0]}{i[1]}{i[2]}')
```
# table의 존재 확인
```
cursor.execute("select count(*) from sqlite_master where name = 'feed'")
    print(cursor.fetchone())
    if(cursor.fetchone()==0):
        cursor.execute("create table feed(title text, content text, Writer text)")
```
