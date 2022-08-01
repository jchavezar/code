#FROM nvidia/cuda:11.2.0-base-ubuntu18.04 
FROM nvidia/cuda:11.2.0-base-ubuntu20.04
RUN apt update && \
     apt install python3 -y && \
     apt install python3-pip -y && \
     python3 -m pip install --upgrade pip && \
     python3 -m pip install gcsfs && \
     pip install -U kfp && \
     pip install tensorflow
