docker build Docker -t ubuntu-torch
docker run -it --ipc=host --gpus 1 -v ~/UniversalGNNs/:/UniversalGNNs/ -v ~/EnergyTransitionTasks/:/EnergyTransitionTasks/ ubuntu-torch

