import random as R

#1부터 10 사이의 자연수 생성 후 x에 저장
x = R.randint(1,10)
print("1과 10 사이의 숫자 생성이 완료되었습니다.")

#게임을 다시시작할지 여부를 묻는 함수
def IsCheck_Regame() :
    print("게임을 다시 진행하시겠습니까?(Y/N) : ")
    str = input()
    if str in ["Y","y"] : return True
    elif str in ["N","n"] : return False
    else :
        print("Y 또는 N 을 입력해 선택해주세요.")
        IsCheck_Regame()

#반복문 시작 (정답 입력 시 종료)
while True :
    n = 0
    try :
        # 0으로 초기화된 변수 n에 플레이어의 예상숫자 저장
        print("예상하는 숫자를 입력하세요.") 
        n = int(input()) 

        #입력값의 범위가 [1,10] 을 벗어나는 경우 재입력 유도
        if n < 1 or n > 10 : print("예상값은 1과 10 사이에 있어야 합니다.")

        #각 조건에 따라 힌트 또는 정답 문구 출력
        elif n > x : print("예상 숫자보다 더 작은 값입니다.")
        elif n < x : print("예상 숫자보다 더 큰 값입니다.")
        elif n == x :
            print("정답입니다!")
            if IsCheck_Regame() : 
                print("게임을 다시 시작합니다.")
                continue
            else :
                print("게임을 종료합니다.")
                break

    #입력값의 타입이 int가 아닌경우 재입력 유도
    except ValueError :
        print("1부터 10 사이의 자연수만 입력할 수 있습니다.")
        continue
        