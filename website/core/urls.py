from django.urls import path, include
from . import views

urlpatterns = [
    path("", views.index, name="home"),
    path("seiri_page/", include("seiri.urls")),
    path("seiton_page/", include("seiton.urls")),
    path("seiso_page/", include("seiso.urls")),
    path("seiketsu_page/", include("seiketsu.urls")),
    # path("shitsuke_page/", include("shitsuke.urls")),
]
