from django.db import models
from user.models import User


class Message(models.Model):
    content = models.TextField(blank=False, null=False)
    role = models.TextField(blank=False, null=False, default="ai")
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='messages')

    def __str__(self):
        return f'{self.content} with id = {self.id}'