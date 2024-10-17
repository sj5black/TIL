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