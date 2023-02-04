from django.db import models

# Create your models here.

class ImageField(models.Model):
    image = models.ImageField(null=True, blank=True, upload_to="static/")

    def get_absolute_url(self):
        return f'view/{self.pk}'
