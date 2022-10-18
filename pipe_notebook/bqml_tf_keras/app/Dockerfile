
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

COPY . /app
WORKDIR /app
RUN pip install tensorflow
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]

EXPOSE 8080
