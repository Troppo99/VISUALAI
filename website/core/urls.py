from django.urls import path, include
from . import views

urlpatterns = [
    path("", views.index, name="home"),
    path("seiso_page/", include("seiso.urls")),
    path("seiketsu_page/", include("seiketsu.urls")),
]
