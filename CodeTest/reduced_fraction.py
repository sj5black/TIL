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