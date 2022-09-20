FROM python:3.7

WORKDIR /

COPY trainer /trainer
COPY requirements.txt requirements.txt

# Installs hypertune library
RUN pip install -r requirements.txt


# Copies the trainer code to the docker image.
COPY trainer /trainer

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python", "-m", "trainer.task"]
