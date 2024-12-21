from django.shortcuts import render

def users(request):
    return render(request, "users/users.html")

def profile(request, username):
    context = {"username" : username }
    return render(request, "users/profile.html", context)
