from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="home"),
    path("seiri_feed/", views.seiri_feed, name="seiri_feed"),
    # path("show_realtime_seiri/", views.show_realtime_seiri, name="show_realtime_seiri"),
]
