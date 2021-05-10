FROM nginxinc/nginx-unprivileged:1-alpine

COPY ./default.conf /etc/nginx/conf.d/default.conf
COPY ./uwsgi_params /etc/nginx/uwsgi_params

USER root
RUN chmod 666 /etc/nginx/conf.d/default.conf
RUN chmod 666 /etc/nginx/uwsgi_params
USER nginx
