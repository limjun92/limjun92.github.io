# Django

* 서버 역할을 할 수 있는 웹 프레임워크
 
* 프레임워크란, 웹 개발을 하기 위한 도구들을 미리 갖춘 것을 말한다.

* 요청에 대한 처리를 하고 응답해준다

## django의 장점

* 많은 것들이 준비가 되어 있어서 빠르다
* 장고의 보안은 매우 안전하다
* 스케일을 키우기에 좋다
* 다양한 것들을 만들 수 있다

## django를 시작하기 전 가상 환경 세팅

* mkdir project_django
* cd project_django
* pipenv shell

## django 설치

* pipenv install django
가상환경에 장고 설치

## django 프로젝트 설치
* django-admin startproject project_with_django
django-admin startproject 프로젝트명


## django의 project/app
* 장고는 하나의 프로젝트(서비스)와 여러 개의 앱(큰 기능)으로 구성된다

## django 서버 실행 하는법
* cd project_with_django -> python manage.py runserver

## django 서버 실행 확인


## django의 ip와  port는?

* 기본적으로 장고의 ip는 loop back ip인 127.0.0.1(localhost)를 사용한다
* ip 127.0.0.1(localhost)
* port 8000

## 배격지식 - Port 란

* sudo lsof -i -n -P | grep LISTEN
모든 포트확인

## 장고의 전체적인 흐름

client -> django(urls.py -> views.py)
urls.py는 요청을 확인해서 views.py의 함수를 호출한다
views.py 가 중요하다
요청에 따라서 DB에 접근하기도한다
models.py DB
응답을 보내는 방식
templates

# 장고의 전체적인 흐름
* 요청을 받고, 응답을 준다!
  1 요청을 받은 url을 urls.py가 찾는다
  2 urls.py에 연결된 views.py가 찾는다
  2.5 templates를 응답에 담거나, models.py의 데이터를 응답에 담는다
  3 views.py에서 응답을 준다
  
* app을 만들고 project의 settings.py의 INSTALLED_APPS에 추가해준다
* urls.py을 다음과 같이 구현

```python
from django.contrib import admin
from django.urls import path
# first_app에서 views를 import 해준다
from first_app import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('',views.helloworld, name='helloworld'),
    path('login/',views.login, name='login'),
    path('signout/',views.signout, name='signout'),
   
]
```

 
```python
# Create your views here.
from django.shortcuts import render
from django.http import HttpResponse

def helloworld(request) :
    return HttpResponse('Hello world!!!')
def login(request) :
    return HttpResponse('로그인 페이지')
def signout(request) :
    return HttpResponse('잘가')
```

views가 templates를 response로 보내는법
views가 templates에 데이터를 주는법

templates는
