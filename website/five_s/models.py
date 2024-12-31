from django.db import models


class ContopCounting(models.Model):
    timestamp = models.TextField()
    jumlah_contop = models.PositiveIntegerField()
    gambar_terakhir = models.ImageField(upload_to="contop_frames/")

    def __str__(self):
        return f"{self.timestamp} - {self.jumlah_contop} Contop"
