from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="home"),
    path("seiso_feed/", views.seiso_feed, name="seiso_feed"),
    # path("show_realtime_seiso/", views.show_realtime_seiso, name="show_realtime_seiso"),
]
