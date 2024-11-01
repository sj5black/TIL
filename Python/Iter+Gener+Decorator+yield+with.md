### Iterable
 - 하나씩 차례대로 값을 꺼내올 수 있는 객체
 - 리스트, 튜플, 문자열, 딕셔너리 등은 모두 반복 가능한 객체
- 이들은 for 문에서 반복할 수 있으며, 내부적으로는 `__iter__()` 메서드를 통해 이터레이터를 반환
```py
numbers = [1, 2, 3, 4, 5]
for num in numbers:
    print(num)
```

### Iterator
 - 반복 가능한 객체의 요소를 하나씩 꺼내오는 객체
 - `__iter__()` 메서드와 `__next__()` 메서드를 구현해야 하며, `__next__()` 메서드를 호출할 때마다 다음 요소를 반환
 - 더 이상 꺼낼 요소가 없으면 `StopIteration` 예외를 발생
 ```py
 numbers = [1, 2, 3]
iterator = iter(numbers)  # 리스트로부터 이터레이터 생성

print(next(iterator))
b = next(iterator)    # next 함수가 호출될 때마다 1씩 증가
print(b)
print(next(iterator))
print(next(iterator)) # StopIteration 발생
```
```
1
2
3
StopIteration                             Traceback (most recent call last)
Cell In[7], line 8
      6 print(b)
      7 print(next(iterator))
----> 8 print(next(iterator))

StopIteration: 
```

```py
# 이터레이터 구현 예시 (실습 필요)
class MyIterator:
    def __init__(self, data):
        self.data = data
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < len(self.data):
            result = self.data[self.index]
            self.index += 1
            return result
        else:
            raise StopIteration

# 이터레이터 사용
my_iter = MyIterator([1, 2, 3])
for item in my_iter:
    print(item)
```

### Generator
 - 이터레이터를 생성하는 특별한 함수
 - `yield` 키워드를 사용해 값을 하나씩 반환
 - 모든 값을 한꺼번에 메모리에 올리지 않고, **필요할 때마다 값을 생성**
 - [] 대신 ()를 사용하여 제너레이터 생성

⭐ **`yield`의 동작 원리**

    return 과 다르게 yield 를 통해 반환된 함수는 다음 호출 시 알고리즘이 처음부터 시작하지 않고,  
    yield를 통해 마지막으로 반환된 시점부터 다시 시작한다.  
    즉, return이 함수의 종료(end)를 의미한다면, yield는 함수의 일시정지(pause)와 같은 느낌이다.

```py
def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

# 피보나치 수열 생성
for num in fibonacci(10):
    print(num)
```

### Decorator
 - 함수나 메서드를 변경하지 않고, 추가적인 기능을 쉽게 추가할 수 있는 방법
 - 함수를 다른 함수로 감싸서, 원래 함수에 새로운 기능을 덧붙일 때 사용

```py
# 데코레이터 사용 예시
def log_decorator(func):
    def wrapper(*args, **kwargs):
        print(f"실행 전: {func.__name__}")
        result = func(*args, **kwargs)
        print(f"실행 후: {func.__name__}")
        return result
    return wrapper

@log_decorator
def say_hello(name):
    print(f"안녕하세요, {name}님!")

# 함수 호출
say_hello("Alice")
```
- 데코레이터 체이닝 시, 안쪽에서부터 밖으로 차례대로 적용
```py
@decorator1
@decorator2
def my_function():
    pass
## 2번 적용 후 1번 적용
```

### with 구문 (Context Manager)
 - 리소스를 획득하고 사용한 뒤, 자동으로 정리해주는 메커니즘
 - 파일을 열고 나서 자동으로 닫거나, 데이터베이스 연결을 관리하는 데 사용
 - `__enter__()`와 `__exit__()` 메서드를 사용
 - `__enter__()` : `with` 블록에 진입할 때 호출. 필요한 리소스를 준비하거나, 설정 작업을 수행
- `__exit__()` : `with` 블록이 끝날 때 호출.
리소스를 정리하고 예외 처리를 수행합니다.

```py
# 컨텍스트 매니저 사용 예시
class MyContextManager:
    def __enter__(self):
        print("리소스를 획득합니다.")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        print("리소스를 정리합니다.")

# 컨텍스트 매니저 사용
with MyContextManager():
    print("작업 수행 중...")
```