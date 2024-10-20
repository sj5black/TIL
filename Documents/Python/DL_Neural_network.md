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