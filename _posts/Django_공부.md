# 환경설정

pipenv shell  
pipenv install django  
  
django-admin startproject 프로젝트명  

cd 프로젝트명

python manage.py startapp 앱이름  
  
setting.py에 app생성을 알려준다  

# 기본

```
# urls.py
from app import views
path('', views.home, name='home')
```

```
# views.py
def home(request):
  return render(requser, 'home.html')
```

```
# app 폴더안에 templates를 생성
  * 하위에 home.html 생성
```

```
# 확인
python manage.py runserver
```

# css 폴더 적용

```
# migrations, templates위치에 static/css/home.css을 만든다

#home.css

h1 {
    color: red;
}

#home.html

<link rel="stylesheet" type="text/css" href="{% static 'css/home.css' %}"/>

# settings.py

STATIC_URL = '/static/'

확인
```


---------------------

* views.py에서 html파일로 데이터 넘기기

```
# views.py
def home(request):
  chat = 'Hello'
  name = 'jun'
  return render(request, 'home.html', {'user_chat':chat, 'user_name':name})
```
```
# home.html
<h1>{{user_chat}}</h1>
<h2>my name is {{user_name}}</h2>
```

* html에서 view로 데이터 넘기기
