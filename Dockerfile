FROM python:latest
WORKDIR /project_emojify
COPY . /project_emojify
RUN pip3 install -r requirements.txt
EXPOSE 3000
CMD ["python","./emojify_dockeflask.py"]