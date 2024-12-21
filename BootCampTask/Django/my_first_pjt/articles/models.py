# DB 동기화 방법(migration)
# 1. python manage.py makemigrations
# 2. python manage.py migrate
"""

"""

from django.db import models

class Article(models.Model):
    title = models.CharField(max_length=50)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.title
