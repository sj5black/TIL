## IDE란? (통합 개발 환경, Integrated Development Environment)
IDE는 프로그래밍을 더 편하게 할 수 있도록 도와주는 소프트웨어
(여러 도구를 하나로 통합해서 제공)
 - 코드 작성 - 자동 완성, 구문 강조 등 편리한 기능을 제공
 - 디버깅 - 코드를 실행하면서 오류를 찾아내고 수정
 - 컴파일/실행 - 코드를 작성하고 바로 실행
1. VSCode (초보자용)
    - MS에서 만든 무료 코드 편집기
    - 확장성: 다양한 확장 기능을 설치해 자신만의 개발 환경 구축
    - 다양한 언어 지원: 파이썬, 자바스크립트, C++ 등 여러 언어 지원
    - 통합 터미널: 별도의 터미널 프로그램 없이, VSCode 내에서 명령어를 실행
&nbsp;
2. PyCharm (전문가용)
    - JetBrains에서 만든 파이썬 전용 IDE (파이썬 특화)
&nbsp;
3. Jupyter Notebook (데이터 분석용)
    - 데이터 과학이나 머신러닝에서 많이 사용되는 도구

__인터랙티브 환경__
코드, 설명, 데이터 시각화 결과를 하나의 문서로 작성할 수 있습니다.<br>
셀 기반 실행 : 코드를 셀 단위로 실행해, 바로바로 결과를 확인할 수 있습니다.<br>
Markdown 지원 : 코드와 설명을 함께 작성할 수 있습니다.<br>

&nbsp;

---

## 변수와 연산자
**1. 변수**
```python
문자 (A-Z, a-z), 숫자 (0-9), 밑줄 (_)**만 사용 가능
숫자로 시작 불가 - 1st_place (❌), first_place (⭕) 
모든 대소문자 구분 (Num, num)
파이썬의 예약어는 변수로 사용 불가 - for, if, class 등
동시 값 할당 가능 (a, b, c = 1, 2, 3)
같은 값 할당 가능 ( x = y = z = 100)
```

**2. 변수의 범위(Scope)**
><small>변수는 어디에서 선언되었는지에 따라 접근할 수 있는 범위가 달라지는데, 이 범위를 **scope**라고 한다. 변수의 스코프는 크게 두 가지로 나눌 수 있다</small>
<small>
 - 전역 변수 (Global Variable) : 프로그램 전체에서 접근할 수 있는 변수. <br>
 - 지역 변수 (Local Variable) : 특정 코드 블록이나 함수 내에서만 접근할 수 있는 변수. </small>
 
**3. 연산자**
```python
2 ** 3 = 8 (거듭제곱),  7 % 3 = 1 (나머지), 7 // 3 = 2 (몫)

x/=2 : 나눈 후 할당 (x = x/2)
x%=2 : 나머지 구한 후 할당 (x = x%2)
x//=2 : 몫을 구한 후 할당 (x = x//2)
x**=2 : 거듭제곱 후 할당 (x = x**2)
``` 

**4. 비트 연산자**
```python
a = 5  # 이진수로 101
b = 3  # 이진수로 011

print(a & b)  # 1 (이진수 001)
print(a | b)  # 7 (이진수 111)
print(a ^ b)  # 6 (이진수 110)
print(~a)     # -6 (이진수 보수)
print(a << 1) # 10 (이진수 1010)
print(a >> 1) # 2 (이진수 010)
```
**5. 보수 연산**
```python
(1) 맨 앞 비트는 부호비트여서 -1의 표현은 1111로 시작한다. 
(2) 양수 5의 표현 
(3) 음수 -6의 표현 (5의 보수)
```
<small>(1) -8 +4 +2 +1 = -1  </small><br>
<small>(2) 0101 = 0×2^3^ + 1×2^2^ + 0×2^1^ + 1×2^0^  </small><br>
<small>(3) 1010= (-1)×2^3^ + 0×2^2^ + 1×2^1^ + 0×2^0^  </small><br>

**6. 멤버십 연산자**
```python
fruits = ["apple", "banana", "cherry"]

print("apple" in fruits)  # True
print("grape" not in fruits)  # True
```
**7. 식별 연산자**
```python
x = ["apple", "banana"]
y = ["apple", "banana"]
z = x

print(x is z)       # True (z는 x를 가리킴)
print(x is y)       # False (x와 y는 내용은 같지만, 다른 객체)
print(x == y)       # True (x와 y는 값이 동일함)
print(x is not y)   # True (x와 y는 다른 객체)
```
&nbsp;

---
## 스트링

```python
# 여러 줄의 문자입력
message = """안녕하세요!
여러 줄의 문자열을 이렇게 작성할 수 있습니다."""

# 부호 연산 가능
ull_name = "Alice" + " " + "Smith"  # Alice Smith
repeated_greeting = "Hello! " * 3    # Hello! Hello! Hello!

# 인덱싱/슬라이싱
text = "Python"
print(text[0])   # 'P'
print(text[1:4]) # 'yth'

# 변수/수식 포함 기능
name = "Teddy"
age = 23
f"{name.upper()} is {age} years old."
>>>  'TEDDY is 23 years old.'
```


### TMI
```python
문자열 내 문자 수정 : str.replace("A","a") 
문자열 내 문자 분리 : str.split(",")
문자열 공백 제거 : str.strip()
문자열 앞 3자리 가져오기 : str[:3]
문자열 뒤 4자리 가져오기 : str[-4:]
문자열 숫자체크 : str.isdigit()
문자열 대소문자 변경 : str.lower() / str.upper()
조건 체크 : len(str) in (4,6) # str의 길이가 4 or 6?
```
---
# 컬렉션 자료형
### 1. 리스트 (List)
 - 여러 개의 항목을 순서대로 저장할 수 있는 **가변 자료형**
 - 대괄호[ ]로 표현하며, 각 요소는 쉼표로 구분
 - 순서가 있다 (Indexing).
 - 가변적이다 (Mutable).
 - 중복된 요소를 가질 수 있다.
```python
fruits = ["apple", "banana", "cherry"]
print(fruits[0])  # apple
print(fruits[-1])  # cherry
print(len(fruits))  # 3
print(fruits.index('apple'))  # 0

fruits[1] = "blueberry"
print(fruits)  # ['apple', 'blueberry', 'cherry']

fruits.append("orange")
print(fruits)  # ['apple', 'blueberry', 'cherry', 'orange']

fruits.extend(["melon", "grapes"])
print(fruits)  # ['apple', 'blueberry', 'cherry', 'orange', 'melon', 'grapes']

fruits.remove("blueberry")
print(fruits)  # ['apple', 'cherry', 'orange', 'melon', 'grapes']

fruits.insert(2, "peach")
print(fruits)  # ['apple', 'cherry', 'peach', 'orange', 'melon', 'grapes']

del fruits[1:4]
print(fruits)  # ['apple', 'melon', 'grapes']

fruits[1:1] = ['mango', 'mango']
print(fruits)  # ['apple', 'mango', 'mango', 'melon', 'grapes']
print(fruits.count('mango'))  # 2

fruits.pop(2)
print(fruits)  # ['apple', 'mango', 'melon', 'grapes']

fruits.reverse()
print(fruits)  # ['grapes', 'melon', 'mango', 'apple']

fruits.sort()
print(fruits)  # ['apple', 'grapes', 'mango', 'melon']

fruits.sort(reverse=True) # r,T 대소문자 주의!
print(fruits)  # ['melon', 'mango', 'grapes', 'apple']
print(bool(fruits))  # True

fruits.clear()
print(fruits)  # []
print(bool(fruits))  # False

a = fruits  # assign (a is fruits) // a 수정 -> fruits도 수정
a = fruits.copy()  # copy (a is not fruits) // a 수정 -> fruits는 유지

"""Variable Expressions in List"""

a = list(i for i in range(10))
print(a)  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

b = list(2*i for i in range(10) if (i%2==0) and (i!=0))
print(b)  # [4, 8, 12, 16]

c = list(i*j for i in range(1,4) for j in range(1,4))
print(c)  # [1, 2, 3, 2, 4, 6, 3, 6, 9]

d = list(map(str, range(5))) # map(int,list) 자체는 포인터 형식
print(d)  # ['0', '1', '2', '3', '4']

e = "".join(d)
print(e)  # 01234

f = list(e)
print(f)  # ['0', '1', '2', '3', '4']
```

### 2. 튜플 (Tuple)
 - 리스트와 유사하지만, 한 번 생성되면 수정할 수 없는 **불변 자료형**
 - 소괄호( )로 표현
 - 순서가 있다 (Indexing).
 - 불변적이다 (Immutable).
 - 중복된 요소를 가질 수 있다.

 
### 3. 딕셔너리 (Dictionary) - 사전
 - 키-값(Key-Value)으로 데이터를 저장하는 자료형
 - 중괄호{ }로 표현되며, 각 키와 값은 콜론으로 구분
 - 순서가 없으며 키를 통해 접근
 - 가변적이다 (Mutable).
 - **키는 유일**해야 하며, **값은 중복**될 수 있다.
```python
# 딕셔너리 생성
person = {"name": "Alice", "age": 25, "city": "New York"}

print(person["name"])  # Alice

# 딕셔너리 값 변경
person["age"] = 26
print(person)  # {'name': 'Alice', 'age': 26, 'city': 'New York'}

# 딕셔너리에 새로운 key-value 추가
person["email"] = "alice@example.com"
print(person)  # {'name': 'Alice', 'age': 26, 'city': 'New York', 'email': 'alice@example.com'}

# 딕셔너리에서 key-value 제거
del person["city"]
print(person)  # {'name': 'Alice', 'age': 26, 'email': 'alice@example.com'}

# 딕셔너리에서 키 목록과 값 목록 접근
print(person.keys())  # dict_keys(['name', 'age', 'email'])
print(person.values())  # dict_values(['Alice', 26, 'alice@example.com'])
```

### 4. 셋 (Set) - 집합
 - 중복되지 않는 요소들의 집합을 나타내는 자료형
 - 중괄호{ }로 표현
 - 순서가 없다 (Unordered).
 - 가변적이다 (Mutable).
 - 중복된 요소를 가질 수 없다.
```python
numbers = {1, 2, 3, 4, 4, 5}  # 중복된 요소는 하나로 처리됨
odd = {1, 3, 5, 7}
even = {2, 4, 6, 8}
print(numbers)  # {1, 2, 3, 4, 5}

numbers.add(6)
print(numbers)  # {1, 2, 3, 4, 5, 6}

numbers.remove(3)
print(numbers)  # {1, 2, 4, 5, 6}

# 합집합(.union)
union_set = odd.union(even)
print(union_set)  # {1, 2, 3, 4, 5, 6, 7, 8}

# 교집합(.intersection)
intersection_set = numbers.intersection(odd)
print(intersection_set)  # {1, 5}
```

**분할 입력**
```python
# 입력하는 데이터들이 문자형 리스트로 a에 저장 (기본 공백 " "으로 구분)
a = input().split()

# 입력하는 데이터들이 문자형 리스트로 a에 저장 (기본 공백 "-"으로 구분)
a = input().split("-")
>> 입력 : 010-xxxx-yyyy
>> 출력 : ["010","xxxx","yyyy"]	split 안의 구분자로 구분

"""Multiple iteration (using zip func)"""

for name, age in zip(names, ages):
    print(f'{name} is {age} years old.')
```