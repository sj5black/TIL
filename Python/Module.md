# 모듈(Module)
 - Python에서 관련된 함수, 클래스, 변수 등을 하나의 파일에 모아놓은 코드 묶음 (Python file)

```python
# nickname (as)
import math as m

result = m.sqrt(25)
print(result)  # 5.0

# choose specific func (doesn't neccessary to add 'm.')
from math import sqrt, pow

result1 = sqrt(49)
result2 = pow(2, 3)
print(result1)  # 7.0
print(result2)  # 8.0

# call all stuffs in math module with *
from math import *  

result = cos(0)
print(result)  # 1.0
```

### 모듈 탐색 경로
Python에서 모듈을 불러올 때, Python은 특정 모듈 탐색 경로에서 해당 모듈을 찾습니다. 일반적으로 Python은 다음 순서로 모듈을 찾습니다:

1. 현재 작업 디렉터리 : 현재 실행 중인 스크립트가 있는 폴더.
2. 표준 라이브러리 경로 : Python이 기본적으로 제공하는 라이브러리들이 위치한 폴더.
3. 환경 변수에 지정된 경로 : PYTHONPATH 환경 변수에 지정된 폴더.

모듈이 이 경로들 중 하나에 존재하면, import 또는 from 구문을 통해 해당 모듈을 불러올 수 있습니다.

 

### 패키지 (Package)
 - 모듈의 모음으로, 여러 모듈을 논리적인 그룹으로 묶은 디렉터리
 - 디렉터리 내에 `__init__.py` 파일이 있어야 Python에서 패키지로 인식
```python
from mypackage import module1
from mypackage.module2 import some_function
```

패키지 설치 : `pip install packagename`  
패키지 목록 확인 : `pip list`  
패키지 업그레이드 : `pip install --upgrade packagename`  
패키지 삭제 :  `pip uninstall packagename`  
패키지 캐시 삭제 (설치 실패 시) : `pip cache purge`  
패키지 특정버전 설치 (설치 실패 시) : `pip install packagename==version no`  
현재 설치된 패키지 목록 기록 : `pip freeze > list.txt`  

**패키지 구조**
```python
mypackage/            # 패키지의 최상위 디렉터리
    __init__.py       # 패키지를 초기화하는 파일 (필수)
    module1.py        # 첫 번째 모듈
    module2.py        # 두 번째 모듈
    subpackage/       # 서브 패키지 (하위 패키지)
        __init__.py   # 서브 패키지 초기화 파일
        submodule.py  # 서브 패키지 내의 모듈
```
&nbsp;
### 가상 환경 (venv)
 - 프로젝트별로 독립된 Python 실행 환경을 만드는 도구
 - 프로젝트마다 서로 다른 패키지 버전을 설치/관리할 수 있어 패키지 간 충돌 방지
```python
가상환경 폴더 생성 : python -m venv 폴더명
가상환경 활성화(Window) : 폴더명\\Scripts\\activate
가상환경 활성화(macOS/Linux) : source 폴더명/bin/activate
```
