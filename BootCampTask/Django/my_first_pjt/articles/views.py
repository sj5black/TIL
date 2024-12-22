"""
장고 공식문서 페이지
https://docs.djangoproject.com/en/4.2/
"""
from django.shortcuts import get_object_or_404, redirect, render
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_POST, require_http_methods
from django.contrib.auth import get_user_model

from .forms import ArticleForm, CommentForm
from .models import Article, Comment
# from django.http import HttpResponse

def index(request):
	# response = HttpResponse("<h1>Hello, Django!</h1>") 
	# return response
    return render(request, "articles/index.html")

@login_required
def create(request):
    form = ArticleForm(request.POST, request.FILES) if request.method == "POST" else ArticleForm()
    
    if request.method == "POST" and form.is_valid():
        article = form.save(commit=False)
        article.author = request.user
        article.save()
        return redirect("articles:article_detail", article.id)

    context = {"form": form}
    return render(request, "articles/create.html", context)

def articles(request):
    articles = Article.objects.all().order_by("-id")
    context = {"articles" : articles}
    return render(request, "articles/articles.html", context)

def article_detail(request, pk):
    article = get_object_or_404(Article, pk=pk)
    comment_form = CommentForm()
    comments = article.comments.all().order_by("-id")
    total_comments = article.comments.count()
    context = {
        "article" : article, 
        "comment_form" : comment_form,
        "comments" : comments,
        "total_comments" : total_comments,
        }
    return render(request, "articles/article_detail.html", context)

@login_required
@require_POST
def delete(request, pk):
    if request.user.is_authenticated:
        article = get_object_or_404(Article, pk=pk)
        if request.user == article.author:
            article.delete()
    return redirect("articles:articles")

@login_required
@require_http_methods(["GET", "POST"])
def update(request, pk):
    article = get_object_or_404(Article, pk=pk)
    if request.user == article.author:
        if request.method == "POST":
            # instance를 선언하면, 새로 만드는게 아닌 기존것을 수정한다
            form = ArticleForm(request.POST, instance=article)
            if form.is_valid():
                article = form.save()
                return redirect("articles:article_detail", article.pk)
        else:
            form = ArticleForm(instance=article)
    else:
        return redirect("articles:articles")
    
    context = {
        "form": form,
        "article": article,
    }
    return render(request, "articles/update.html", context)

@login_required
@require_POST
def comment_create(request, pk):
    article = get_object_or_404(Article, pk=pk)
    form = CommentForm(request.POST)
    if form.is_valid():
        comment = form.save(commit=False)
        comment.article = article
        comment.author = request.user
        comment.save()
    return redirect("articles:article_detail", article.pk)

@login_required
@require_POST
def comment_delete(request, article_pk, comment_pk):
    comment = get_object_or_404(Comment, pk=comment_pk)
    if request.user == comment.author:
        comment.delete()
    return redirect("articles:article_detail", article_pk)

@require_POST
def like(request, article_pk):
    if request.user.is_authenticated:
        article = get_object_or_404(Article, pk=article_pk)
        user = request.user
        if article.like_users.filter(pk=user.pk).exists():
            article.like_users.remove(user)
        else:
            article.like_users.add(user)
        return redirect("articles:articles")
    return redirect("accounts:login")





















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


##### 이전 버전 메서드 ######

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

# def delete(request, pk):
#     article = get_object_or_404(Article, pk=pk)
#     if request.method == "POST":
#         article.delete()
#         return redirect("articles:articles")
#     return redirect("articles:article_detail", article.pk)