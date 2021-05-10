from rest_framework.parsers import MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status

from django.http import Http404
from .models import File
from .serializers import FileSerializer, ResultSerializer
from .apps import UploadwaveConfig
import os
import threading


class WaveFactoryView(APIView):
    parser_classes = (MultiPartParser,)

    # get object from sql with pk
    def get_object(self, pk):
        try:
            return File.objects.get(pk=pk)
        except File.DoesNotExist:
            raise Http404

    # Http Get Method
    # request must has a parameter 'uuid'
    # eg: http://12.345.678.910/wave_factory/?uuid=8cb8f409-ecd5-4eb6-b5c7-cef4900c72cc
    def get(self, request, *args, **kwargs):
#         uuid = request.GET.get('uuid')
#         if uuid is None:
#             return Response(status=status.HTTP_400_BAD_REQUEST)

#         file = self.get_object(uuid)
#         result_serializer = ResultSerializer(data={'uuid': file.uuid, 'result': file.result})

#         if result_serializer.is_valid():
#             return Response(result_serializer.data, status=status.HTTP_200_OK)
#         else:
#             return Response(result_serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        return Response("How are you!!!!!!!", status=status.HTTP_200_OK)

    # Http Post Method
    # request body must has two keys: uuid, file
    def post(self, request, *args, **kwargs):
        file_serializer = FileSerializer(data=request.data)

        if file_serializer.is_valid():
            file_serializer.save()

            file = self.get_object(request.data.get('uuid'))
            file_path = r'media/' + str(file)
            # create a new thread to process wave file
            threading.Thread(target=self.process_data, args=(file_path, request)).start()

            return Response(file_serializer.data, status=status.HTTP_201_CREATED)
        else:
            return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    # Http Delete Method
    # request must has a parameter 'uuid'
    # eg: http://12.345.678.910/wave_factory/?uuid=8cb8f409-ecd5-4eb6-b5c7-cef4900c72cc
    def delete(self, request, *args, **kwargs):
        uuid = request.GET.get('uuid')
        if uuid is None:
            return Response(status=status.HTTP_400_BAD_REQUEST)
        file = self.get_object(uuid)
        file_path = r'media/' + str(file)
        file.delete()
        os.remove(file_path)
        return Response(status=status.HTTP_204_NO_CONTENT)

    # process data with speaker diarization model and save the result into sql
    def process_data(self, path, request):
        result = UploadwaveConfig.predictor.predict(path)
        file = self.get_object(request.data.get('uuid'))
        file.result = result
        file.save()

