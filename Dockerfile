FROM tiangolo/uwsgi-nginx-flask:python3.6

RUN apt-get update && apt-get install -y \
  htop \
  vim \
  less 
RUN pip install numpy
RUN mkdir /var/log/uwsgi

COPY ./app /app