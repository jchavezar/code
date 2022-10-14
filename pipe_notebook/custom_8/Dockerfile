
FROM tensorflow/tensorflow

RUN pip install google-cloud-bigquery==2.34.4 \
    && pip install pandas \
    && pip install protobuf==3.19.4 \
    && pip install pyarrow \
    && pip install cloudml-hypertune

COPY * /

ENTRYPOINT ["python", "main.py"]
