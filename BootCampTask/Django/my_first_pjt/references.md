

```bash
<Django 공식 문서>
https://docs.djangoproject.com/en/4.2/

<터미널 명령어>
conda create -n venv python=3.12.7
pip install django==4.2
pip freeze > requirements.txt
django-admin startproject my_first_pjt
python manage.py startapp accounts >> 이후 settings.py 에 앱 등록
python manage.py runserver
python manage.py createsuperuser
python manage.py changepassword <username>

<Shell 문법으로 DB 수정>
pip install django-extensions >> 이후 settings.py 에 앱 등록
pip install ipython
pip freeze > requirements.txt
python manage.py shell_plus

<DB 활성화>
Ctrl+Shift+P 이후 선택

```

```python
Shell 문법으로 DB 수정하기.

1. 매니저를 통한 Article 객체 생성 (DB 추가)
Article.objects.create(title='third title', content='마지막 방법임')
Article.objects.all()

2. DB 조회 구문 (단일 조회)
Article.objects.get(id=1)
Article.objects.get(content='my_content')

3. 조건 조회
Article.objects.filter(id__gt=2) # 2보다 큰 id 모두 조회
Article.objects.filter(id__in=[1,2,3]) # 1,2,3에 속하는 id 조회
Article.objects.filter(content__contains='my') # my가 포함된 content 조회

4. 특정 DB 수정/삭제
article = Article.objects.get(id=1)
article.title = 'updated title'
article.save() // article.delete()
```