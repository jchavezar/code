
FROM rapidsai/rapidsai-nightly:22.04-cuda11.2-base-ubuntu20.04-py3.9

RUN pip install google.cloud[storage] \
  && pip install gcsfs \
  && pip install xgboost --upgrade

COPY trainer trainer/

ENTRYPOINT ["python", "trainer/task.py"]
