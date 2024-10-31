## Branch와 Revert, Reset

1. Branch (브랜치)

```bash
$ git checkout -b my_branch  # my_branch라는 이름의 브랜치를 만들고 그 브랜치로 이동하기
$ git branch -D my_branch  # my_branch 지우기
$ git branch my_branch  # my_branch 만들기
```

2. Revert (리버트)
 - 이전 코드 기록을 남겨둔 채로 변경 사항을 취소

```bash
$ git revert e23e9e  # e23e93 커밋으로 되돌아가기
```

3. Reset (리셋)
 - 내가 작업한 기록을 아예 없애버리는 것
 - Reset은 기록이 남지 않고 작업이 완전히 없어지기 때문에 주의해서 사용

```bash
$ git reset --hard  # 모든 변경 기록 삭제
$ git reset --soft e23e9e  # e23e9e 커밋은 임시저장 되어있음
$ git reset --mixed e23e93  # e23e9e 커밋 직전으로 돌아가고, 임시저장은 안 되어있음
```
&nbsp;

    git clone : 원격 저장소를 로컬 저장소로 복사하는 명령어
    git clone <원격 저장소 주소>

    git log : 커밋 히스토리를 보여주는 명령어
    git diff : 변경된 내용을 비교하는 명령어
​​
4. GUI
 - Git을 더 쉽게 사용할 수 있는 그래픽 사용자 인터페이스(GUI) 도구
 - SourceTree : 그래픽 인터페이스로 Git을 쉽게 사용할 수 있는 프로그램
 - GitHub Desktop : GitHub와 연결하여 Git을 좀 더 쉽게 다룰 수 있게 도와주는 도구

```bash
## 원격의 feature/login 가져오기 ##

0. 원격 저장소의 최신 정보 동기화 (작업 시작 전 필수)
- git fetch origin

1. 원격 브랜치 확인
- git branch -r  # 원격 브랜치만 보기
- git branch -a  # 로컬과 원격 브랜치 모두 보기

2. 원격 브랜치를 로컬에 가져오기 (두 가지 방법)
# 방법 1 - checkout -b 사용
- git checkout -b feature/login origin/feature/login

# 방법 2 - switch -c 사용 (최신 git 버전)
- git switch -c feature/login origin/feature/login

3. 작업 전 브랜치가 제대로 가져와졌는지 확인
- git branch  # 현재 브랜치 확인
- git status  # 현재 상태 확인

4. 코드 작업 수행
- # (파일 수정, 생성 등)

5. 변경사항 확인
- git status  # 변경된 파일 확인
- git diff    # 변경 내용 상세 확인

6. 커밋하기
- git add .   # 또는 git add 특정파일명
- git commit -m "commit message"

7. 원격 저장소에 푸시
- git push origin feature/login
```