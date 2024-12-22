from django.shortcuts import render, redirect
from django.contrib.auth import login as auth_login
from django.contrib.auth import logout as auth_logout
from django.contrib.auth import update_session_auth_hash
from django.contrib.auth.forms import (
    AuthenticationForm, UserCreationForm, PasswordChangeForm)
from django.views.decorators.http import require_POST, require_http_methods
from .forms import CustomUserChangeForm, CustomUserCreationForm


@require_http_methods(["GET", "POST"])
def login(request):
    print(f"로그인 시 request값 : {request}")
    if request.method == "POST":
        form = AuthenticationForm(data = request.POST)
        if form.is_valid():
            auth_login(request, form.get_user())
            next_url = request.GET.get("next") or "index"
            return redirect(next_url)
        else:
            # 폼이 유효하지 않은 경우 에러 메시지를 템플릿에 전달
            error_message = "Invalid username or password. Please try again."
            context = {"form": form, "error_message": error_message}
            return render(request, "accounts/login.html", context)
    form = AuthenticationForm()
    context = {"form" : form}
    return render(request, "accounts/login.html", context)

@require_POST
def logout(request):
    if request.user.is_authenticated:
        auth_logout(request)
    return redirect("index")

@require_http_methods(["GET", "POST"])
def signup(request):
    if request.method == "POST":
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            auth_login(request, user)
            return redirect("index")
    form = CustomUserCreationForm()
    context = {"form" : form}
    return render(request, "accounts/signup.html", context)

@require_POST
def delete(request):
    print(request)
    if request.user.is_authenticated:
        request.user.delete()
        auth_logout(request)
    return redirect("index")

@require_http_methods(["GET", "POST"])
def update(request):
    if request.method == "POST":
        form = CustomUserChangeForm(request.POST, instance=request.user)
        if form.is_valid():
            form.save()
            return redirect("index")
    form = CustomUserChangeForm(instance=request.user)
    context = {"form" : form}
    return render(request, "accounts/update.html", context)

def change_password(request):
    if request.method == "POST":
        form = PasswordChangeForm(request.user, request.POST)
        if form.is_valid():
            form.save()
            update_session_auth_hash(request, form.user)
            return redirect("index")
    form = PasswordChangeForm(request.user)
    context = {"form" : form}
    return render(request, "accounts/change_password.html", context)
    
    