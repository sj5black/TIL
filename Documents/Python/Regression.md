# 회귀 모델(Regression Model)

### 머신 러닝
**1. 훈련 데이터 (Training Data)**
 - 가중치 업데이트
  모델의 가중치값은 훈련 데이터를 사용하여 업데이트됩니다. 훈련 과정에서 모델은 훈련 데이터를 통해 패턴을 학습하고, 손실 함수를 최소화하기 위해 가중치를 조정합니다.
 - 학습 과정
 모델은 훈련 데이터를 사용하여 입력과 출력 사이의 관계를 학습합니다. 이 과정에서 모델은 가중치를 업데이트하여 예측의 정확도를 향상시킵니다.

**2. 테스트 데이터 (Test Data)**
 - 성능 평가
  테스트 데이터는 모델을 학습하는 데 사용되지 않으며, 모델의 일반화 능력을 평가하는 데 사용됩니다. 즉, 훈련 데이터에 대해 학습한 모델이 실제 데이터(테스트 데이터)에서 얼마나 잘 작동하는지를 측정하는 데 사용됩니다.
 - 가중치 변화 없음
  테스트 데이터는 모델의 가중치에 영향을 미치지 않습니다. 모델이 훈련을 통해 이미 설정된 가중치를 가지고 있을 때, 테스트 데이터로 예측을 수행하고 결과를 평가합니다.

**3. 모델 평가의 중요성**
 - 일반화 능력 평가
  테스트 데이터는 모델이 새로운, 보지 않은 데이터에 대해 얼마나 잘 일반화되는지를 평가하는 데 중요한 역할을 합니다. 모델이 훈련 데이터에만 잘 맞춰져 있다면(즉, 오버피팅), 테스트 데이터에서 성능이 저하될 수 있습니다.
 - 훈련/검증/테스트 분할
  일반적으로 데이터셋은 훈련 데이터, 검증 데이터, 테스트 데이터로 분할됩니다. 검증 데이터는 하이퍼파라미터 조정 및 모델 선택을 위한 중간 평가를 제공하고, 테스트 데이터는 최종 성능을 평가하는 데 사용됩니다.

### 선형 회귀 (선형)
 - 종속 변수와 하나 이상의 독립 변수 간의 선형 관계를 모델링 하는 방법


$$y=β_0​+β_1​x_1​+ϵ$$ $$\scriptsize\textsf{ 단순 선형 회귀}$$ <br>
$$y=β_0​+β_1​x_1​+β_2​x_2​+⋯+β_n​x_n​+ϵ$$ $$\scriptsize\textsf{ 다중 선형 회귀}$$ 

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 데이터 생성
X = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5],[6,6]])
y = np.array([1, 2, 3, 4, 5, 6])

# 데이터 분할 (훈련 데이터와 테스트 데이터)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# random_state 의 값은 난수의 초기값을 의미하므로, 설정을 하는 것 자체에 의미를 둔다. 값은 중요X

# 선형 회귀 모델 생성 및 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)
a = list(map(float,y_pred))
print(a)  # [1.0000000000000013, 2.000000000000001]

# 모델 평가 (y_test는 정답, y_pred는 모델의 예측값)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')  # 테스트 데이터에 대한 MSE 출력
print(f'R^2 Score: {r2}')  # 테스트 데이터에 대한 R² Score 출력
```
**KeyPoint**
`random_state`의 역할은 주어진 `X(특징)`와 `Y(타겟)` 데이터 셋의 조합을 바꾸는 것이 아니라, `train data`와 `test data`를 랜덤하게 나누는 과정에서 무작위성을 부여하는 것이다.
&nbsp;
### 다항 회귀 (비선형 feature, 선형 model)
  - 종속 변수와 독립 변수 간의 비선형 관계를 모델링하는 방법
  - 다항회귀 차수(degree) : 독립 변수의 최대 차수(n)
  - 차수가 높을수록 모델이 더 복잡해지며 과적합(overfitting)의 위험 → 적절한 차수 선택 필요

$$y=β_0​+β_1​x+β_2​x^2+⋯+β_n​x^n+ϵ$$

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 데이터 생성
X = np.array([[1], [2], [3], [4], [5], [6]])
y = np.array([1, 4, 9, 16, 25, 36])

# 다항 특징 생성 (차수 2)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# 데이터 분할 (훈련 데이터와 테스트 데이터)
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)
# random_state 의 값은 난수의 초기값을 의미하므로, 설정을 하는 것 자체에 의미를 둔다. 값은 크게 중요X

# 다항 회귀 모델 생성 및 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 모델 평가
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')  # 테스트 데이터에 대한 MSE 출력
print(f'R^2 Score: {r2}')  # 테스트 데이터에 대한 R² Score 출력
```
**KeyPoint**
`X_poly`에 포함되는 변수들은 기존 `X`의 0차~n차 항 만큼의 변수들이 추가되어 포함된다.

&nbsp;
### 리지 회귀 (선형)
- 회귀 계수의 크기를 제어하여 과적합을 방지하는 정규화 기법
- L2 정규화(regularization)를 사용하여 회귀 계수의 제곱합을 최소화
- $\lambda$ 는 정규화 강도를 조절하는 hyper_parameter

$$J(\beta) = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{p} \beta_j^2$$


```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

# 데이터 생성
X = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]])
y = np.array([1, 2, 3, 4, 5, 6])

# 데이터 분할 (훈련 데이터와 테스트 데이터)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 리지 회귀 모델 생성 및 학습
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 모델 평가
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')  # 테스트 데이터에 대한 MSE 출력
print(f'R^2 Score: {r2}')  # 테스트 데이터에 대한 R² Score 출력
```
&nbsp;
### 라쏘 회귀 (선형)
- 회귀 계수의 크기를 제어하여 과적합을 방지하는 정규화 기법
- L1 정규화(regularization)를 사용하여 회귀 계수의 절대값 합을 최소화
- $\lambda$ 는 정규화 강도를 조절하는 hyper_parameter

$$J(\beta) = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{p} |\beta_j|$$

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score

# 데이터 생성
X = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6,6]])
y = np.array([1, 2, 3, 4, 5, 6])

# 데이터 분할 (훈련 데이터와 테스트 데이터)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 라쏘 회귀 모델 생성 및 학습
model = Lasso(alpha=1.0)
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 모델 평가
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')  # 테스트 데이터에 대한 MSE 출력
print(f'R^2 Score: {r2}')  # 테스트 데이터에 대한 R² Score 출력
```

**L1 정규화**

- 일부 회귀 계수를 0으로 만들어 특징 선택(feature selection)을 수행
- 모델의 해석 가능성을 높이고, 불필요한 특징을 제거하는 데 유용

**L2 정규화**

- 모든 가중치를 작게 만들어 모델의 복잡도 축소
- 손실 함수에 제곱항을 추가하여 매끄러운 최적화 가능
&nbsp;
### 로지스틱 회귀
- 종속 변수가 이진형일 때(결과값이 2개일 때) 사용되는 통계 기법
- 선형 회귀와 달리 결과값이 0과 1 사이에 위치하게 하기 위해 시그모이드 함수(Sigmoid Function)를 사용
- 각 데이터 포인트가 특정 클래스에 속할 확률을 예측
- $z=β_0​+β_1​x+β_2​x^2+⋯+β_n​x^n$ 인 `z`에 대해,

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**비용 함수**
<small>
- 모델의 예측 확률과 실제 레이블 사이의 차이를 측정
- Log Loss function 또는 Cross-Entropy loss function 으로 불림
</small>

$$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(h_\theta(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)})) \right]$$