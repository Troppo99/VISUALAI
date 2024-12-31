# urls.py
from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="home"),
    path("seiketsu_feed/<str:camera_name>/", views.seiketsu_feed, name="seiketsu_feed"),
    # path("show_realtime_seiketsu/", views.show_realtime_seiketsu, name="show_realtime_seiketsu"),
]
