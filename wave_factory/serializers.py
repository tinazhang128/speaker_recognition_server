from rest_framework import serializers
from .models import File


class FileSerializer(serializers.ModelSerializer):
    class Meta:
        model = File
        # fields = '__all__'
        exclude = ['result']


class ResultSerializer(serializers.Serializer):
    uuid = serializers.CharField()
    result = serializers.JSONField()
