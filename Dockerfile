FROM tiangolo/uwsgi-nginx-flask:python3.6

RUN apt-get update && apt-get install -y \
  apt-utils \
  htop \
  vim \
  less
RUN pip install numpy

COPY ./app /app