from django.urls import path
from . import views # 현재 위치의 views 임포트

app_name = "articles"
urlpatterns = [
    path("", views.articles, name="articles"),
    path("create/", views.create, name="create"),
    path("<int:pk>/", views.article_detail, name="article_detail"),
    path("<int:pk>/delete/", views.delete, name="delete"),
    path("<int:pk>/update/", views.update, name="update"),

    path("hello/", views.hello, name="hello"),
    path("data-throw/", views.data_throw, name="data-throw"),
    path("data-catch/", views.data_catch, name="data-catch"),
]