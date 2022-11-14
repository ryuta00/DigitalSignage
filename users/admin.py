from django.contrib import admin
from .models import News, Reaction, Student, StudentReaction


admin.site.register(News)
admin.site.register(Reaction)
admin.site.register(Student)
admin.site.register(StudentReaction)
