import uuid
from django.db import models


# Map to the 'file' database table
class File(models.Model):
    uuid = models.CharField(primary_key=True, max_length=36)
    file = models.FileField(blank=False, null=False)
    result = models.TextField(default='null')

    def __str__(self):
        return self.file.name
