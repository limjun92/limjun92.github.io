# position

## :static 
default 값
-위에서 적용된 position 값을 초기화 시키고 싶을때 사용

## :relative 
static 이였을때의 위치를 기준으로 이동

## :absolute
* position static이 아닌 부모가 있을 때 

      부모를 기준으로 움직임

* position static이 아닌 부모가 없을 때

      body를 기준으로 움직임

## :fixed

* 스크롤을 무시한다
  
## :sticky

* 스크롤시 부딪치는 순간 고정


# diplay 
* 화면에 요소를 어떻게 표시할 지를 선택하는 속성

## :inline

* 다음 요소를 줄바꿈하지 않는다
* width, height 속성이 무시된다

## :block

* inline content 인 요소를 block으로 바꿀 수 있다

## :none;

* 요소를 화면에 표시하지 않음

## :inline-block
* inline과 같지만, width, height속성 사용 가능

# display:flex;
* 요소를 자유자재로 위치 시킬 수 있다
* Container와 Item으로 이루어져 있다

## flex-direction
* items의 주축(main-axis)를 바꾼다

```css
flex-direction:row; /* default */
flex-direction:column;
```

## justify-content
* main-axis의 정렬 방법을 정의한다

```css
justify-content:flex-start; /* default */
justify-content:flex-end;
justify-content:center;
justify-content:space-between;
justify-content:space-around;
```

## align-item
* cross-axis의 정렬 방법을 정의

```css
align-items:stretch; /*default*/
align-items:flex-start;
align-items:center;
align-items:flex-end;
```
