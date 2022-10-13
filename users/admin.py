from django.contrib import admin
from .models import News
from .models import Reaction


admin.site.register(News)
admin.site.register(Reaction)
