FROM python:3.8-slim

COPY scripts /home/workdir
COPY requirements.txt /home/workdir

RUN apt-get update && apt-get install -y git

RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu


RUN pip install mlflow==1.25.1
# --no-cache-dir mlflow
RUN pip install scikit-learn
# --no-cache-dir scikit-learn

RUN pip install -r /home/workdir/requirements.txt

ENTRYPOINT [ "/bin/bash" ]