FROM nvcr.io/nvidia/pytorch:24.03-py3

WORKDIR /workspace/Wav2Vec-Wrapper/

ENV HF_HOME="/workspace"

COPY . .

RUN pip3 install -r requirements.txt
RUN pip3 uninstall -y transformer-engine && pip3 install transformers==4.39.3
