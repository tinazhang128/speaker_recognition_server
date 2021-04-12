FROM tiangolo/uwsgi-nginx:python3.7
COPY . /app
WORKDIR /app
RUN apt-get update
RUN apt-get install python-pyaudio portaudio19-dev python3-pyaudio -y
RUN apt-get install libsndfile1-dev -y
RUN pip install -r requirement.txt
EXPOSE 80