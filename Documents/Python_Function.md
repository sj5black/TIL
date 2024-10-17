# 함수
**기본 생성 구조**
```python
def func(param1, param2):
    # code
    return return_value  # return이 정의되지 않으면 None 반환
```
### enumerate(list)
- 각 자료값에 index를 부여하여 tuple 로 병합
```python
numbers = ["안", "녕", "하", "세", "요"]
f_numbers = enumerate(numbers)
for i,j in f_numbers :
    print(i,j)
```

### items()
 - dictionary 형태의 자료형을 tuple로 묶어 list 형태로 변환
```python
a = {'name' : "Jace", 'age' : 23, 'address' : "LA" }

b = a.items()
print(b)  # dict_items([('name', 'Jace'), ('age', 23), ('address', 'LA')])
```

### 타입 변환 함수
```python
int(data)    # data를 int형으로 변환
float(data)  # data를 float형으로 변환
str(data)    # data를 string형으로 변환
bool(data)   # data를 boolean형으로 변환
list(data)   # data를 list형으로 변환
tuple(data)  # data를 tuple형으로 변환
set(data)    # data를 set형으로 변환
```

### 기타 내장 함수
 - Python에서 기본적으로 제공하는 함수
 - 별도의 모듈을 임포트(import)하지 않아도 언제든지 사용
```python
type(data)  # data의 타입을 반환
len(data)   # data 요소의 개수를 반환
sum(data)   # data 요소의 합을 반환
min(data)   # data 요소의 최소값을 반환
max(data)   # data 요소의 최대값을 반환
abs(data)   # data 의 절대값을 반환
round(data, digit)   # data 를 소수점 아래 digit까지 반올림하여 반환
input("Message : ")  # 입력받은 값을 문자열로 반환
```
&nbsp;
**다중 기본값 (Multiple Default Parameter)**
<small>
 - 함수는 여러 매개변수에 대해 기본값을 가질 수 있다.
 - 기본값이 있는 매개변수는 항상 기본값이 없는 매개변수 뒤에 와야한다.
</small>
```python
def order(item, quantity=1, price=1000):
```
**복수 반환값**
<small>
  - 여러 개의 반환값을 반환하는 경우 **튜플** 형태로 반환
</small>
```python
def calculate(a, b):
    s = a + b
    difference = a - b
    return s, difference

c, d = calculate(10, 5)
print(c)  # 15
print(d)  # 5
```

**가변 매개변수 (Variable-Length Arguments)**

 - 정해지지 않은 개수의 인수를 받고 싶을 때 사용
 - `*args`와 `*kwargs` 의 특정 매개변수 명칭으로 구분
 - `*args`와 `**kwargs`를 함께 사용할 때에는 `*args`를 먼저, `**kwargs`를 나중에 정의
 ```python
 # *args (tuple type)
 def add(*args):
    return sum(args)

print(add(1, 2, 3))         # 6
print(add(10, 20, 30, 40))  # 100

# **kwargs (dic type)
def print_info(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}")

print_info(name="Alice", age=30, city="Seoul")
# name: Alice
# age: 30
# city: Seoul
 ```