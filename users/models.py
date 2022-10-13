from django.db import models
from django.contrib.auth.models import User


class News(models.Model):
    text = models.CharField(max_length=500)


class Reaction(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    news = models.ForeignKey(News, on_delete=models.CASCADE)
    stamp = models.IntegerField()
