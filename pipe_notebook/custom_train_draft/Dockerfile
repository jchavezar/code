
FROM python:latest

RUN pip install pandas \
    && pip install gcsfs \
    && pip install fsspec

COPY trainer/ /trainer

ENTRYPOINT ["python", "-m", "trainer.train"]
