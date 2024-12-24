from django.urls import path, include
from . import views

urlpatterns = [
    path("", views.index, name="home"),
    path("seiso_page/", include("seiso.urls")),
    path("seiketsu_page/", include("seiketsu.urls")),
    path("seiketsu_feed/", views.seiketsu_feed, name="seiketsu_feed"),
    # path("show_realtime_seiketsu/", views.show_realtime_seiketsu, name="show_realtime_seiketsu"),
]
