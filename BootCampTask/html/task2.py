#Person 클래스 생성
class Person() :
    #생성자 선언 : name, gender, age를 매개변수로 받아 객체가 생성될 때 각 속성값들이 초기화되도록 설정
    def __init__(self, name, gender, age):
        
        #입력받은 성별이 male 또는 female이 아닐 경우, 올바른 입력이 될때까지 재입력 유도
        while gender not in ['male', 'female'] :
            print("잘못된 성별입니다. 'male' 또는 'female'을 입력하세요.")
            gender=(input('성별 : '))

        self.name = name
        self.gender = gender
        self.age = age

    #display 메서드 생성 : 객체(self)의 속성값들을 호출하여 출력
    def display(self) :
        print(f'이름 : {self.name}, 성별 : {self.gender}')
        print(f'나이 : {self.age}')

    #greet 메서드 생성 : 객체(self)의 나이에 따른 인사말 분리
    def greet(self) :
        if self.age > 60 : print('어서오세요 어르신. 환영합니다.')
        elif self.age > 19 : print(f'{self.name}! 성인이시군요! 반갑습니다.')
        elif self.age > 13 : print(f'{self.name}! 청소년이시군요! 반갑습니다.')
        elif self.age <= 13 : print(f'안녕, {self.name}! 어린이구나! 반가워~')

#나이 입력 함수. 정수 변환 시 발생할 수 있는 ValueError의 원활한 예외처리 및 진행을 위해 생성
def input_age() : 
    while True : 
        age_temp = input('나이 : ')
        try : 
            result = int(age_temp)
            #나이가 1보다 작아도 루프문 상단으로 반환
            if result < 1 :
                print("잘못된 값입니다. 1 이상의 정수를 입력하세요.")
            #모든 유효성 검사 종료 후 정상적인 나이값 반환(무한루프 종료)
            return result
        except ValueError : 
            print("잘못된 값입니다. 1 이상의 정수를 입력하세요.")

#입력받을 나이, 이름, 성별에 대한 변수 선언 및 초기화
Age = 0
Name = ''
Gender = ''

#human1의 요소(Atrribute) 입력
Age = input_age()
Name = input('이름 : ')
Gender = input('성별 : ')

#human1이라는 객체 생성
human1 = Person(Name, Gender, Age)

#human1이라는 객체의 display, greet 메서드 실행
human1.display()
human1.greet()

