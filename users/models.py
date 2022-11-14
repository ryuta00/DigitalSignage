from django.db import models
from django.contrib.auth.models import User


class News(models.Model):
    title = models.CharField(max_length=100)
    text = models.CharField(max_length=500)


class Reaction(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    news = models.ForeignKey(News, on_delete=models.CASCADE)
    stamp = models.IntegerField()  # ok:1 or no:0


class Student(models.Model):
    name = models.CharField(max_length=20)
    mail = models.EmailField()
    vector_path = models.CharField(max_length=100)


class StudentReaction(models.Model):
    user = models.ForeignKey(Student, on_delete=models.CASCADE)
    news = models.ForeignKey(News, on_delete=models.CASCADE)
    stamp = models.IntegerField()  # ok:1 or no:0
