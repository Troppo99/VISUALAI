from django.contrib import admin
from django.urls import path
from .views import streaming_index, camera_feed

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", streaming_index, name="home"),
    path("camera_feed/<str:camera_name>/", camera_feed, name="camera_feed"),
]
