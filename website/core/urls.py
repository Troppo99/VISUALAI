from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="home"),
    # path('seiri/', views.seiri, name='seiri')
    # (Jika nanti ingin menambahkan halaman full reload, kita bisa tambahkan)
]
