FROM python:3.8-slim-buster

COPY /classifier /classifier
COPY logging.conf.yml .
RUN pip --no-cache-dir install Flask

CMD ["python3", "classifier/flask_api.py"]