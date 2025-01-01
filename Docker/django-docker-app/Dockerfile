# (베이스 이미지)
# python:3.9-slim이라는 이미지를 Docker hub에서 가져와 줘!
FROM python:3.9-slim

# (작업 디렉토리 설정)
# 앞으로 돌아갈 컨테이너(프로그램)에는 app이라는 폴더를 만들어서, 거기에서 작업할 거야
WORKDIR /app

# (종속성 설치)
# requirements.txt을 WORKDIR로 설정한 app이라는 폴더에 옮겨줘.
COPY requirements.txt .

# requirements.txt에 있는 거 전부 설치해줘.
RUN pip install --no-cache-dir -r requirements.txt


# (소스 코드 복사)
# 지금 폴더를 컨테이너에 다 복사해줘.
COPY . .

# (포트 노출)
# 8000번 포트만 컨테이너에서 외부에 노출(expose)시킬 거야.
EXPOSE 8000

# (명령 실행)
# python manage.py runserver 0.0.0.0:8000 명령어를 컨테이너에서 실행해줘.
# 0.0.0.0은 외부에 연결 가능한 것을 의미
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]


# 터미널 명령어

# 가상환경 생성(일반)
# python -m venv myvenv_1234

# 가상환경 활성화 (현재 폴더에 가상환경 있는 경우)
# myvenv_1234\\Scripts\\activate 또는
# .\myvenv_1234\Scripts\activate

# docker야, 지금 경로(.)를 django-app이라는 이름(-t)으로 빌드(build)해줘.
# docker build -t django-app .

# django-app이라는 컨테이너를 컨테이너로 실행(run)해줘. 
# 내 컴퓨터에서 8000 포트로 접속하면, 컨테이너의 8000 포트로 연결해줘.(-p 8000:8000)
# 대신에, 너는 터미널에 붙어있지 말아줘. (-d)
# docker run -d -p 8000:8000 django-app

# 컨테이너 목록 보기
# docker ps

# 컨테이너 중지
# docker stop 66399f228574(컨테이너 ID)

# 컨테이너 삭제
# docker rm 66399f228574(컨테이너 ID)

