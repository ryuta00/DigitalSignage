from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required

from users.models import News, Reaction


def login(request):
    return render(request, 'login.html')


@login_required
def home(request):
    return render(request, 'home.html')


@login_required
def news(request):
    if request.method == 'POST':
        Reaction.objects.create(
            user=request.user,
            stamp=request.POST['stamp'],
            news=News.objects.get(id=request.POST['news'])
        )
        return redirect("news")

    else:
        ctx = {
            "news_list": News.objects.all()
        }
        return render(request, 'news.html', ctx)
