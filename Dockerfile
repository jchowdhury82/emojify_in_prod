FROM python:3.6-slim-buster
MAINTAINER joyjitchowdhury
WORKDIR /emojify
COPY . /emojify
RUN pip3 install -r requirements.txt
CMD ["python","emojify_flask.py"]