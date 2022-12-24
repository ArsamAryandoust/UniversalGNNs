FROM ubuntu:latest

ENV https_proxy http://proxy.ethz.ch:3128/
ENV http_proxy http://proxy.ethz.ch:3128/

RUN apt-get update && apt-get upgrade -y
RUN apt-get install python3-pip -y

RUN pip3 install torch torchvision torchaudio
RUN pip3 install pandas
RUN pip3 install tqdm

WORKDIR /UniversalGNNs
