from django.contrib import admin
from django.urls import path
from . import views, consumers
from django.urls import re_path

urlpatterns = [
    path('ws/video_stream/', views.video_stream, name='video_stream'),
]

websocket_urlpatterns = [
    re_path(r'ws/video_stream/$', consumers.VideoStreamConsumer),
]
