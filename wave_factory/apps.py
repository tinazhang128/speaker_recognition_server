from django.apps import AppConfig
from .model.speakerUtil import Predictor


class UploadwaveConfig(AppConfig):
    name = 'wave_factory'

    predictor = Predictor()
    predictor.load_model()
