from django.urls import path
from . import views # 현재 위치의 views 임포트

urlpatterns = [
    path("hello/", views.hello, name="hello"),
    path("data-throw/", views.data_throw, name="data-throw"),
    path("data-catch/", views.data_catch, name="data-catch"),
]