FROM python:3.7
COPY . /app
WORKDIR /app
EXPOSE 8000
RUN apt-get update
RUN apt-get install python-pyaudio portaudio19-dev python3-pyaudio -y
RUN apt-get install libsndfile1-dev -y
RUN pip3 install --no-cache-dir -r requirement.txt
ENTRYPOINT [ "python" ]
CMD [ "manage.py", "runserver", "0.0.0.0:8000" ]
