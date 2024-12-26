from django.contrib import admin
from .models import ContopCounting


@admin.register(ContopCounting)
class ContopCountingAdmin(admin.ModelAdmin):
    list_display = ("id", "timestamp", "jumlah_contop")
    readonly_fields = ("timestamp",)
    list_filter = ("timestamp", "jumlah_contop")
    search_fields = ("id",)
