# Class
- 속성(Attributes)과 메서드(Methods)로 구성


**속성(Attributes)**
클래스에 정의된 변수로, 객체의 데이터를 정의

**메서드(Methods)**
클래스에 정의된 변수로, 객체의 동작을 정의

**객체(Object)**
클래스의 인스턴스(Instance)로, 클래스라는 설계도로부터 실제로 생성된 구체적인 데이터와 기능을 가지는 실체


    Dog : 클래스  
    my_dog : 객체  
    name, breed : 속성   
    bark() : 메서드

 

### 클래스 만들기

```python
class Dog:
    pass
```
- pass 구문은 class 이름만 정의하고 내용을 비워둘 경우 사용


1. 생성자 정의
```python
def __init__(self, name, breed) :
    self.name = name
    self.breed = breed
```
 - 속성은 클래스의 생성자 메서드인 `__init__` 에서 정의
 - `__init__` 메서드는 객체가 생성될 때 자동으로 호출되며, 객체의 초기 속성을 설정

2. 메서드 정의
```py
    def bark(self):
        return f"{self.name}가 짖습니다."

    def introduce(self):
        return f"이름: {self.name}, 품종: {self.breed}"
```

3. 객체 생성
```py
# 여러 개의 Dog 객체 생성
dog1 = Dog("Max", "Bulldog")
dog2 = Dog("Bella", "Poodle")
```
---
## 매직 메서드
 -  Python에서 특별한 역할을 수행하는 **미리 정의된** 메서드
 ```py
 ```