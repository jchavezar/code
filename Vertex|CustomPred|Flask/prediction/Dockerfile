FROM python:3.7-buster

RUN mkdir my-model

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt 

COPY app.py ./app.py

# Flask Env Variable
ENV FLASK_APP=app

# Expose port 8080
EXPOSE 8080

CMD flask run --host=0.0.0.0 --port=8080