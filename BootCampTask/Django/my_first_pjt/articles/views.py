"""
장고 공식문서 페이지
https://docs.djangoproject.com/en/4.2/
"""
from django.shortcuts import redirect, render

from .forms import ArticleForm
from .models import Article
# from django.http import HttpResponse

def index(request):
	# response = HttpResponse("<h1>Hello, Django!</h1>") 
	# return response
    return render(request, "articles/index.html")

def create(request):
    form = ArticleForm(request.POST) if request.method == "POST" else ArticleForm()
    
    if request.method == "POST" and form.is_valid():
        article = form.save()
        return redirect("articles:article_detail", article.id)

    context = {"form": form}
    return render(request, "articles/create.html", context)

def articles(request):
    articles = Article.objects.all().order_by("-id")
    context = {"articles" : articles}
    return render(request, "articles/articles.html", context)

def article_detail(request, pk):
    article = Article.objects.get(id=pk)
    context = {"article" : article}
    return render(request, "articles/article_detail.html", context)

def delete(request, pk):
    article = Article.objects.get(pk=pk)
    if request.method == "POST":
        article.delete()
        return redirect("articles:articles")
    return redirect("articles:article_detail", article.pk)

def update(request, pk):
    article = Article.objects.get(pk=pk)
    
    if request.method == "POST":
        # instance를 선언하면, 새로 만드는게 아닌 기존것을 수정한다다
        form = ArticleForm(request.POST, instance=article)
        if form.is_valid():
            article = form.save()
            return redirect("articles:article_detail", article.pk)
    else:
        form = ArticleForm(instance=article)

    context = {
        "form": form,
        "article": article,
    }
    return render(request, "articles/update.html", context)

# def edit(request, pk):
#     article = Article.objects.get(pk=pk)
#     context = {"article": article}
#     return render(request, "articles/edit.html", context)

# def new(request):
#     form = ArticleForm()
#     context = {"form": form}
#     return render(request, "articles/new.html", context)

# def create(request):
#     title = request.POST.get("title")
#     content = request.POST.get("content")
#     article = Article.objects.create(title=title, content=content)
#     return redirect("articles:article_detail", article.pk)

# def update(request, pk):
#     article = Article.objects.get(pk=pk)
#     article.title = request.POST.get("title")
#     article.content = request.POST.get("content")
#     article.save()
#     return redirect("articles:article_detail", article.pk)

def data_throw(request):
    return render(request, "articles/data_throw.html")

def data_catch(request):
    message = request.GET.get("message")
    context = {"data" : message, }
    return render(request, "articles/data_catch.html", context)

def hello(request):
    name = "Teddy"
    tags = ["python", "django", "html", "css"]
    books = ["해변의 카프카", "코스모스", "백설공주", "어린왕자"]

    context = {
        "name" : name,
        "tags" : tags,
        "books" : books,
        }
    return render(request, "articles/hello.html", context)
