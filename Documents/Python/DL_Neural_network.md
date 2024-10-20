## 퍼셉트론
 - 인공 신경망의 가장 기본적인 단위로, 하나의 뉴런을 모델링한 것
 - 입력 값을 받아 가중치(weight)를 곱하고, 이를 모두 더한 후 활성화 함수(activation function)를 통해 출력 값을 결정

 <img src="./images/PNN.png" style="width:40%; height:auto;display: block; margin: 0 auto;">

 $$ y = f(\sum_{i=1}^{n} w_i x_i + b) $$

 여기서 $x_i$는 입력 값, $w_i$는 가중치, $b$는 bias, $f$는 활성화 함수입니다.

## 다층 퍼셉트론 (MLP)
- 여러 층의 퍼셉트론을 쌓아 올린 신경망 구조
- 입력층(input layer), 은닉층(hidden layer), 출력층(output layer)으로 구성

<img src="./images/MLP.png" style="width:45%; height:auto;display: block; margin: 0 auto;">

- **입력 레이어(Input Layer) :** 외부 데이터가 신경망에 입력되는 부분. 입력 레이어의 뉴런 수는 입력 데이터의 특징 수와 동일
- **은닉 레이어(Hidden Layer) :** 입력 데이터 처리, 특징 추출 
- **출력 레이어(Output Layer) :** 최종 예측 값을 출력. 출력 레이어의 뉴런 수는 예측하려는 클래스 수 또는 회귀 문제의 출력 차원과 동일

## 활성화 함수 (Activation function)
 - 신경망의 각 뉴런에서 입력값을 출력값으로 변환
 - 단순 선형변환에서 벗어나 비 선형성을 도입하여 신경망이 복잡한 패턴을 학습하도록 유도
 &nbsp;
 1. **ReLU** **(Rectified Linear Unit)**
    
    $$
    f(x) = \max(0, x)
    $$
    
    - 장점: 계산이 간단하고, 기울기 소실 문제(vanishing gradient problem) 완화
    - 단점: 음수 입력에 대해 기울기가 0이 되는 '죽은 ReLU' 문제 발생 가능
&nbsp;
2. **Sigmoid**
    
    $$
    f(x) = \frac{1}{1 + e^{-x}} 
    $$
    
    - 장점: 출력 값이 0과 1 사이로 제한되어 확률을 표현하기에 적합
    - 단점: 기울기 소실 문제, 출력 값이 0 또는 1에 가까워질 때 학습이 느려지는 단점
&nbsp;
3. **Tanh (Hyperbolic Tangent)**
    
    $$
    f(x) = \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
    $$
    
    - 장점: 출력 값이 -1과 1 사이로 제한
    - 단점: 기울기 소실 문제 발생 가능

    ---
# 손실 함수와 알고리즘
### 손실 함수 (loss funciton)
- 모델의 예측 값과 실제 값 사이의 차이를 측정하는 함수
- 모델의 성능을 평가하고, 최적화 알고리즘을 통해 모델을 학습시키는데 사용  
&nbsp;
1. **MSE (Mean Squared Error)** - 회귀 문제에서 주로 사용

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 
$$



2. **Cross-Entropy** - 분류 문제에서 주로 사용 (예측 확률과 실제 클래스 간의 차이를 측정)

$$
\text{Cross-Entropy} = -\sum_{i=1}^{n} y_i \log(\hat{y}_i)
$$

### 최적화 알고리즘 (Optimization Algorithm)
- 손실 함수를 최소화하기 위해 모델의 가중치를 조정
- 손실 함수의 기울기를 계산하고, 이를 바탕으로 가중치를 업데이트

### 역전파 알고리즘 (Backpropagation)
- 신경망의 가중치를 학습시키기 위해 사용되는 알고리즘
- 출력에서 입력 방향으로 손실 함수의 기울기를 계산하고, 이를 바탕으로 가중치를 업데이트

**역전파의 수학적 원리**
- 연쇄 법칙(Chain Rule)을 사용해 손실함수의 기울기 계산
- 각 층의 기울기는 이전 층의 기울기와 현재 층의 기울기를 곱하여 계산
- 이를 통해 신경망의 모든 가중치 업데이트

<img src="./images/Backpro.png" style="width:50%; height:auto;display: block; margin: 0 auto;">

---
# 인공 신경망 (ANN)