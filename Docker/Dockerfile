FROM ubuntu:latest

#ENV https_proxy http://proxy.ethz.ch:3128/
#ENV http_proxy http://proxy.ethz.ch:3128/

RUN apt-get update && apt-get upgrade -y
RUN apt-get install python3-pip -y

RUN pip3 install torch torchvision torchaudio
RUN pip3 install pandas
RUN pip3 install tqdm
RUN pip3 install scikit-learn
RUN pip3 install pytorch-lightning
# install torch_geometric
RUN pip3 install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
RUN pip3 install matplotlib
ENV PATH="${PATH}:/UniversalGNNs/src:/UniversalGNNs/src/datasets"

WORKDIR /UniversalGNNs
