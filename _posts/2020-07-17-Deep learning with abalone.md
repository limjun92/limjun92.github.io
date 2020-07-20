---
title:  "Deep learning with abalone"
excerpt: "Deep learning 알고리즘을 활용하여 전복의 고리 수 추정"
toc: true
toc_sticky: true
categories:
  - AI_개념
tags:
  - AI
  - Deep learning
last_modified_at: 2020-07-17
---

```python
import numpy as np
import csv
import time
import os
```

```python
RND_MEAN = 0
RND_STD = 0.0030

LEARNING_RATE = 0.001
```

# 메인함수

```python
def abalone_exec(epoch_count = 10, mb_size = 100, report = 1):
    print(epoch_count,mb_size,report)
    load_abalone_dataset()
    # 데이터을 읽어오고 가공
    init_model()
    # 가중치와 편향을 초기화
    train_and_test(epoch_count, mb_size, report)
    # 훈련과 테스트
```

# 데이터 적재 및 가공

```python
def load_abalone_dataset():
    os.chdir(r"C:\Users\user\limjun92.github.io\data\인공지능사관학교\deep")
    with open('abalone.csv') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader,None) # 가장 처음 행을 제거해 준다
        rows = []
        #print(type(csvreader))
        #print(type(rows))
        for row in csvreader:
            rows.append(row) 
            # _csv.reader을 list로 변환

    global data, input_cnt, output_cnt 
    # 전역 변수는 미리 선언하고 사용

    input_cnt, output_cnt = 10,1
    data = np.zeros([len(rows),input_cnt+output_cnt]) 
    #print(len(rows))
    #행렬을 만들고 0을 입력
    # print(data)
    for n, row in enumerate(rows):
        if row[0] == 'I': data[n,0] = 1
        if row[0] == 'M': data[n,1] = 1
        if row[0] == 'F': data[n,2] = 1
        # one_hot 0,1,2 를 원-핫 벡터를 사용해구현한다
        data[n,3:] = row[1:]
        # 나머지 데이터는 바로 뒤에 붙여준다
    #print(data)
```

# 파라미터 초기화 함수 정의 

```python
def init_model():
    global weight, bias, input_cnt, output_cnt
    weight = np.random.normal(RND_MEAN, RND_STD,[input_cnt, output_cnt])
    # 평균이 0이고 표준편차 0.003 인 랜덤값을 weight로 만들어준다
    #print(weight)
    bias = np.zeros([output_cnt])
    # 편향?
    #print(bias)
```

# 학습 및 평가 함수 정의

```python
def train_and_test(epoch_count, mb_size, report):
    step_count = arrange_data(mb_size)
    # 전복의 데이터 수는 4177개
    # 80%의 데이터를 훈련으로 사용함
    # mb_size가100일떄 step_count는 33이다
    test_x, test_y = get_test_data()
    # test를 위해 사용하는 data
    # test_x는 전복의 정보
    # test_y는 전복의 고리수
    
    # train_data와 test_data가 고정된다
    
    for epoch in range(epoch_count):
        #epoch_count = 100
        losses, accs = [],[]
        for n in range(step_count):
            #step_count = 33
            train_x, train_y = get_train_data(mb_size,n)
            # n이 0일때 get_train_data에서 섞어주기 때문에 epoch가 증가 할때마다 매번 다른값을 가져온다
            # 3300개의 데이터를 100개씩 쪼개서 33번 가져온다
            loss, acc = run_train(train_x, train_y)
            #losses.append(loss)
            #accs.append(acc)
            losses.append(loss)
            #오차제곱의 평균을 입력
            accs.append(acc)
            #정확도 입력
        if report > 0 and (epoch+1) % report == 0:
            acc = run_test(test_x, test_y)
            print('Epoch {}: loss={:5.3f}, accuracy={:5.3f}/{:5.3f}'. \
                  format(epoch+1, np.mean(losses), np.mean(accs), acc))
            
    final_acc = run_test(test_x, test_y)
    print('\nFinal Test: final accuracy = {:5.3f}'.format(final_acc))
```

## train_data와 test_data를 분할

```python
def arrange_data(mb_size):
    global data, shuffle_map, test_begin_idx
    shuffle_map = np.arange(data.shape[0])
    # data.shape[0]을 통해 행의 수를 알수 있다
    # data.shape[1]을 통해 열의 수를 알수 있다
    # np.arange(N) 는 N수만큼 array형태로 반환해주는 함수이다
    #print(shuffle_map)
    np.random.shuffle(shuffle_map)
    # shuffle_map을 random으로 섞어준다
    #print(shuffle_map)
    step_count = int(data.shape[0] * 0.8) // mb_size
    # 80%의 데이터를 훈련으로 사용
    #print(step_count) = 33
    test_begin_idx = step_count*mb_size
    # 4176의 자료 중에서 test를 시작하는 idx(3300부터 시작)
    #print(test_begin_idx)
    return step_count
```

## test를 위해 사용하는 data 정의

```python
def get_test_data():
    global data, shuffle_map, test_begin_idx, output_cnt
    test_data = data[shuffle_map[test_begin_idx:]]
    # 섞여있는 shuffle_map의 test_begin_idx(3300)부터 끝까지 shuffle_map의 값을
    # index로 가지고 있는 data를 가져온다
    #print(test_data)
    #print(test_data[:, :-output_cnt])
    # 전복의 정보
    #print(test_data[:, -output_cnt:])
    # 전복의 고리수
    return test_data[:, :-output_cnt], test_data[:, -output_cnt:]
```

## train를 위해 사용하는 data 정의

```python
def get_train_data(mb_size, nth):
    global data, shuffle_map, test_begin_idx, output_cnt
    if nth == 0:
        #print(shuffle_map[:test_begin_idx])
        np.random.shuffle(shuffle_map[:test_begin_idx])
        #print(shuffle_map[:test_begin_idx])
        # 각 epoch를 시작할때마다 train_data를 섞어준다
    train_data = data[shuffle_map[mb_size*nth:mb_size*(nth+1)]]
    # mb_size(100) 만큼 return해준다
    return train_data[:,:-output_cnt], train_data[:,-output_cnt:]
```

# train 알고리즘

```python
def run_train(x, y):
    # x는 data, y는 그에 대한 이미 알고 있는 결과
    output, aux_nn = forword_neuralnet(x)
    # 가중치와 편향을 적용
    loss, aux_pp = forward_postproc(output, y)
    # 오차 제곱의 평균과 오차행렬
    accuracy = eval_accuracy(output, y)
    # print(accuracy)
#     G_loss = 1.0
#     G_output = backprop_postproc(G_loss, aux_pp)
#     backprop_neuralnet(G_output, aux_nn)
    G_loss = 1.0
    
    G_output = backprop_postproc(G_loss,aux_pp)
    backprop_neuralnet(G_output, aux_nn)
    
    return loss, accuracy
```

## 순전파

```python
def forward_neuralnet(x):
    # x는 훈련을 위한 데이터
    global weight, bias
    output = np.matmul(x,weight) + bias
    # matmul은 2차원에서 dot과 동일하나 3차원이상에서는 다른 결과를 보인다
    # dot은 내적곱인 반면 matmul은 두 배열의 행렬곱이다
    # 가중치를 곱한후 편향을 더해준다
    #print(weight.shape[0],weight.shape[1])
    #print(x.shape[0], x.shape[1])
    return output, x
```

## 오차구하기

```python
def forward_postproc(output, y):
    # output은 neuralnet 결과
    # y는 실제 결과값
    #print(type(output),output.shape[0],output.shape[1])
    #print(type(y),y.shape[0],y.shape[1])
    
    diff = output - y
    # 계산된 결과값과 실제 결과값의 차를 구한다
    square = np.square(diff)
    # 계산된 결과값과 실제 결과값의 차를 제곱한다
    loss = np.mean(square)
    # 계산된 결과값과 실제 결과값의 차를 제곱하고 평균을 구한다
    
    return loss,diff
    # loss는 오차 제곱의 평균 
    # diff는 오차 행렬
```

## 정확도 

```python
def eval_accuracy(output, y):
    #print(output[0])
    #print(y[0])
    midff = np.mean(np.abs((output-y)/y))
    return 1-midff
```

## 가중치 와 편향에 변화를 주기위한 인자 구하기

```python
def backprop_postproc(G_loss, diff):
    #======================print(diff[0:5])
    # diff 는 실제 결과값과 출력 결과값의 오차를 행렬로 저장한것
    shape = diff.shape
    #print(shape)
    #print(type(shape))
    
    g_loss_square = np.ones(shape) / np.prod(shape)
    #print(np.prod(shape))
    #print(np.ones(shape))
    #print(type(shape))
    #print(type(np.ones(shape)))
    #np.prod(N) N배열안의 모든 값을 곱한다
    #np.ones(shape) shape크기의 행렬을 만들고 모든값을 1로 초기화 한다
    #print((g_loss_square.shape))
    #np.prod(shape) = 100 일경우 g_loss_square는 모든값이 0.01인 (100,1) 행렬
    
    g_square_diff = 2 * diff
    # diff값을 2배해서 g_square_diff에 입력
    g_diff_output = 1
    
    G_square = g_loss_square * G_loss
    #print(G_square)
    #모든값이 0.01로 초기화된 행렬 * G_loss
    G_diff = g_square_diff * G_square
    #오차 행렬의 두배 * G_square
    G_output = g_diff_output * G_diff
    #g_diff_output(초기화 1) * G_diff
    
    #========================print(G_output[0:5])
    
    return G_output
    # 함수에서 diff를 입력으로 받아서 G_output으로 변환한후 return 
    # 값이 작아졌다
```

## 가중치 와 편향에 변화 주기

```python
def backprop_neuralnet(G_output, x):
    #x = data
    global weight, bias
    #print(x.shape)
    g_output_w = x.transpose()
    # transpose 전치행렬을 만든다, x의 행과 열을 바꾸어 g_output_w에 저장한다.
    #print(g_output_w.shape)
    
    G_w = np.matmul(g_output_w, G_output)
    #print(weight)
    #print(G_w)
    
    G_b = np.sum(G_output,axis = 0)
    #print(G_b)
    
    #print(weight)
    weight -= LEARNING_RATE * G_w
    # 가중치 조정
    #print(weight)
    #print(bias)
    bias -= LEARNING_RATE * G_b
    # 편향 조정
    #print(bias)
```

# test 구현

```python
def run_test(x, y):
    output, _ = forward_neuralnet(x)
    accuracy = eval_accuracy(output, y)
    return accuracy
```

# 출력

```python
abalone_exec()
```
10 100 1  
Epoch 1: loss=11.023, accuracy=0.780/0.799  
Epoch 2: loss=6.874, accuracy=0.812/0.795  
Epoch 3: loss=6.729, accuracy=0.811/0.809  
Epoch 4: loss=6.588, accuracy=0.813/0.806  
Epoch 5: loss=6.525, accuracy=0.813/0.809  
Epoch 6: loss=6.438, accuracy=0.815/0.798  
Epoch 7: loss=6.367, accuracy=0.815/0.816  
Epoch 8: loss=6.293, accuracy=0.817/0.814  
Epoch 9: loss=6.223, accuracy=0.819/0.801  
Epoch 10: loss=6.170, accuracy=0.819/0.809  

Final Test: final accuracy = 0.809

## 가중치와 편향의 변화 값

```python
print(weight)
print(bias)
```

[[  0.39626569]  
 [  1.39064843]  
 [  1.26638338]  
 [  4.32336958]  
 [  5.18607575]  
 [  4.5062529 ]  
 [  5.03768753]  
 [-16.05903141]  
 [ -2.49417499]  
 [ 12.0969977 ]]  
[3.04648438]  

# 하이퍼퍼라미터 수정하며 실험
* LEARNING_RATE = 0.001 -> 0.1
* epoch_count = 10 -> 100
* mb_size = 10 -> 100

```python
LEARNING_RATE = 0.1
abalone_exec(epoch_count=100,mb_size=100,report=20)
```
100 100 20  
Epoch 20: loss=5.905, accuracy=0.824/0.834  
Epoch 40: loss=5.378, accuracy=0.831/0.841  
Epoch 60: loss=5.189, accuracy=0.835/0.838  
Epoch 80: loss=5.119, accuracy=0.837/0.841  
Epoch 100: loss=5.082, accuracy=0.837/0.843  
  
Final Test: final accuracy = 0.843  

# 새로운 입력 벡터 X에 대한 예측

```python
x = np.array([0,1,0,0.44,0.3,0.08,0.5,0.23,0.11,0.2])
output = forward_neuralnet(x)
print(output)
```
(array([9.22604521]), array([0.  , 1.  , 0.  , 0.44, 0.3 , 0.08, 0.5 , 0.23, 0.11, 0.2 ]))
