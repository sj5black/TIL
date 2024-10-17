# 회귀 모델(Regression Model)

### 선형 회귀 (선형)
 - 종속 변수와 하나 이상의 독립 변수 간의 선형 관계를 모델링 하는 방법


$y=β_0​+β_1​x_1​+ϵ$ $\scriptsize\textsf{ (단순 선형 회귀)}$ <br>
$y=β_0​+β_1​x_1​+β_2​x_2​+⋯+β_n​x_n​+ϵ$ $\scriptsize\textsf{ (다중 선형 회귀)}$ 

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

# 선형 회귀 모델 생성 및 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 모델 평가
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')
```
&nbsp;
### 다항 회귀 (비선형)
  - 종속 변수와 독립 변수 간의 비선형 관계를 모델링하는 방법
  - 다항회귀 차수(degree) : 독립 변수의 최대 차수(n)
  - 차수가 높을수록 모델이 더 복잡해지며 과적합(overfitting)의 위험 → 적절한 차수 선택 필요

$y=β_0​+β_1​x+β_2​x^2+⋯+β_n​x^n+ϵ$

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

# 다항 회귀 모델 생성 및 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 모델 평가
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')
```
&nbsp;
### 리지 회귀 (선형)
- 회귀 계수의 크기를 제어하여 과적합을 방지하는 정규화 기법
- L2 정규화(regularization)를 사용하여 회귀 계수의 제곱합을 최소화
- $\lambda$ 는 정규화 강도를 조절하는 hyper_parameter

$ J(\beta) = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{p} \beta_j^2 $


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
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')
```
&nbsp;
### 라쏘 회귀 (선형)
- 회귀 계수의 크기를 제어하여 과적합을 방지하는 정규화 기법
- L1 정규화(regularization)를 사용하여 회귀 계수의 절대값 합을 최소화
- $\lambda$ 는 정규화 강도를 조절하는 hyper_parameter

$ J(\beta) = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{p} |\beta_j| $

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
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')
```

---
### **L1 정규화**

- 일부 회귀 계수를 0으로 만들어 특징 선택(feature selection)을 수행
- 모델의 해석 가능성을 높이고, 불필요한 특징을 제거하는 데 유용

### **L2 정규화**

- 모든 가중치를 작게 만들어 모델의 복잡도 축소
- 손실 함수에 제곱항을 추가하여 매끄러운 최적화 가능