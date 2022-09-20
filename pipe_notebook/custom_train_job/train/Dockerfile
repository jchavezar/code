
FROM python:latest

RUN python -m pip install --upgrade pip
RUN pip install pandas gcsfs tensorflow
COPY / /trainer

CMD ["python", "trainer/main.py"]
