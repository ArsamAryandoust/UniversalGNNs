FROM ubuntu:latest

RUN apt-get update && apt-get upgrade -y
RUN apt-get install python3-pip -y

RUN pip3 install torch torchvision torchaudio
RUN pip3 install jupyter
RUN pip3 install pandas
RUN pip3 install h5py

WORKDIR /UniversalGNNs

CMD jupyter notebook --ip 0.0.0.0 --port 1111 --no-browser --allow-root
