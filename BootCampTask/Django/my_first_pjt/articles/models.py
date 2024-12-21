# DB 동기화 방법(migration)
# 1. python manage.py makemigrations
# 2. python manage.py migrate
"""
Shell 문법으로 DB 수정하기.

1. 관련라이브러리 설치
pip install django-extensions
pip install ipython

2. settings.py 에 앱 등록
"django_extensions",

3. shell_plus로 실행
python manage.py shell_plus

4. 매니저를 통한 Article 객체 생성 (DB 추가)
Article.objects.create(title='third title', content='마지막 방법임')
Article.objects.all()

5. DB 조회 구문 (단일 조회)
Article.objects.get(id=1)
Article.objects.get(content='my_content')

6. 조건 조회
Article.objects.filter(id__gt=2) # 2보다 큰 id 모두 조회
Article.objects.filter(id__in=[1,2,3]) # 1,2,3에 속하는 id 조회
Article.objects.filter(content__contains='my') # my가 포함된 content 조회

7. 특정 DB 수정/삭제
article = Article.objects.get(id=1)
article.title = 'updated title'
article.save() // article.delete()
"""

from django.db import models

class Article(models.Model):
    title = models.CharField(max_length=50)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.title
