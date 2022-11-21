from django.urls import path, include
from django.contrib.auth import views as auth_views
from django.conf import settings
from django.conf.urls.static import static
from . import views

urlpatterns = [
    path('login/', views.login, name='login'),
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),
    path('social-auth/', include('social_django.urls', namespace='social')),  # Googleログイン
    path('news/<news>/', views.news, name='news'),  # ニュースを表示, 顔認証
    path('news/<news>/<user>/', views.reaction, name='reaction'),  # ニュースを表示, リアクションを取得
    path('start/', views.start, name='start'),  # 初期設定(初期データを登録), テータをリセットした時に行う
    path('match/<news>/', views.match, name='match'),  # 顔認証によるログイン
    path('camera/<news>/<user>/', views.camera, name='camera'),  # どちらの手を挙げているか調べる
    path('signup/', views.signup, name='signup'),  # 写真をアップロードして生徒登録
    path("", views.home, name='home'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
