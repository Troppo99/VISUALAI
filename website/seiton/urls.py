from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="home"),
    path("seiton_feed/", views.seiton_feed, name="seiton_feed"),
    # path("show_realtime_seiton/", views.show_realtime_seiton, name="show_realtime_seiton"),
]
