
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

COPY / /app
WORKDIR /app
RUN python -m pip install --upgrade pip
RUN pip install pandas gcsfs tensorflow
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]

EXPOSE 8080
