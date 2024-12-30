from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from . import views

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", views.index, name="index"),
    path("seiri_page/", include("seiri.urls")),
    path("seiton_page/", include("seiton.urls")),
    path("seiso_page/", include("seiso.urls")),
    path("seiketsu_page/", include("seiketsu.urls")),
    # path("shitsuke_page/", include("shitsuke.urls")),
]


if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
