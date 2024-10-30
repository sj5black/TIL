문제 설명
첫 번째 분수의 분자와 분모를 뜻하는 numer1, denom1, 두 번째 분수의 분자와 분모를 뜻하는 numer2, denom2가 매개변수로 주어집니다. 두 분수를 더한 값을 기약 분수로 나타냈을 때 분자와 분모를 순서대로 담은 배열을 return 하도록 solution 함수를 완성해보세요.

제한사항
0 <numer1, denom1, numer2, denom2 < 1,000

입출력 예 #1  
1 / 2 + 3 / 4 = 5 / 4입니다. 따라서 [5, 4]를 return 합니다.  

입출력 예 #2  
9 / 2 + 1 / 3 = 29 / 6입니다. 따라서 [29, 6]을 return 합니다.

---
```python
import math

def solution(numer1, denom1, numer2, denom2):
    answer = []
    denominator = int((denom1*denom2)/math.gcd(denom1, denom2))
    numerator = int((numer1*denominator/denom1) + (numer2*denominator/denom2))
    GCD = math.gcd(denominator, numerator)

    #약분
    answer+=[int(numerator/GCD), int(denominator/GCD)]

    return answer

print(solution(5,8,7,12))
```