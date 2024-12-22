
<!-- ! + TAB 시 html 기본포맷 자동완성 -->

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
python manage.py collectstatic
 >> settings.py 에 아래 구문 추가
    STATICFILES_DIRS = [BASE_DIR / "static"]
    STATIC_ROOT = BASE_DIR / "static/"

<DB 구조에 변경이 있을 때(model 수정 시)>
1. python manage.py makemigrations
2. python manage.py migrate

<Shell 문법으로 DB 수정>
pip install django-extensions >> 이후 settings.py 에 앱 등록
pip install ipython
pip freeze > requirements.txt
python manage.py shell_plus

<DB 활성화>
Ctrl+Shift+P 이후 선택

<이미지 처리 패키지 설치>
pip install pillow
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

5_1. 코멘트 추가
python manage.py shell_plus
article = Article.objects.get(pk=14)
Comment.objects.create(content="first commit", article=article)

5_2. 코멘트 추가
python manage.py shell_plus
article = Article.objects.get(pk=14)
comment = Comment()
comment.article = article
comment.content = "second commit"
comment.save()

6. article -> comment 역참조 (_set 사용 or related name 으로 별명 설정)
article.comment_set.all()
```