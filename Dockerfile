FROM tiangolo/uwsgi-nginx-flask:python2.7

RUN pip install pandas sklearn scipy
RUN mkdir /var/log/uwsgi

RUN apt-get update && apt-get install -y \
  htop vim less \
  && rm -rf /var/lib/apt/lists/*

COPY ./app/appserver.py /appserver.py
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

ENTRYPOINT ["/entrypoint.sh"]

# Add demo app
COPY ./app /app
WORKDIR /app

CMD ["/usr/bin/supervisord"]
