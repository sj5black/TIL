"""
장고 공식문서 페이지
https://docs.djangoproject.com/en/4.2/
"""
from django.shortcuts import render
# from django.http import HttpResponse

def index(request):
	# response = HttpResponse("<h1>Hello, Django!</h1>") 
	# return response
    return render(request, "index.html")

def hello(request):
    name = "Teddy"
    tags = ["python", "django", "html", "css"]
    books = ["해변의 카프카", "코스모스", "백설공주", "어린왕자"]

    context = {
        "name" : name,
        "tags" : tags,
        "books" : books,
        }
    return render(request, "hello.html", context)

def data_throw(request):
    return render(request, "data_throw.html")

def data_catch(request):
    message = request.GET.get("message")
    context = {"data" : message, }
    return render(request, "data_catch.html", context)

