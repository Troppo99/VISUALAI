from django.db import models


class check(models.Model):
  id = models.AutoField(primary_key=True)
  timestamp = models.DateTimeField(auto_now_add=True)
  message = models.CharField(max_length=100)

  def __str__(self):
    return "{}".format(self.message)